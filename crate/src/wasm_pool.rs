//! Shared-memory worker pool for the threaded wasm artifact (issue #44).
//!
//! Coordinator (main wasm thread inside `embed`) writes a job list, `Release`s
//! an epoch, and `memory_atomic_wait32`s until `workers_done == W`.
//! Workers park on the epoch, compute a disjoint output-column range, then
//! `Release` on `workers_done`.
//!
//! No allocations on the worker path. Weights and the Q8_K tile are immutable
//! for the dispatch; each worker writes only its half-open column range.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::gguf::TensorType;
use crate::qmatmul::{column_range, matmul_ggml_cols, BlockQ8K, QuantMat};

pub(crate) const MAX_SITES: usize = 2;
pub(crate) const TY_Q4K8: u32 = 1;
pub(crate) const TY_Q4K: u32 = 2;
pub(crate) const TY_Q5K: u32 = 3;
pub(crate) const TY_Q6K: u32 = 4;

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct SiteJob {
    pub ty: u32,
    pub n_tokens: u32,
    pub n_in: u32,
    pub n_out: u32,
    pub n_blocks: u32,
    pub align: u32,
    pub w_ptr: u32,
    pub w_len: u32,
    pub q_ptr: u32,
    pub q_len: u32,
    pub y_ptr: u32,
    pub y_len: u32,
}

struct Control {
    epoch: AtomicU32,
    n_sites: AtomicU32,
    n_workers: AtomicU32,
    workers_done: AtomicU32,
    shutdown: AtomicU32,
    sites: UnsafeCell<[SiteJob; MAX_SITES]>,
}

// SiteJob is integer-only. The happens-before is epoch Release/Acquire:
// coordinator writes sites, then fetch_add(epoch, Release); workers
// Acquire epoch before reading sites.
unsafe impl Sync for Control {}

static CONTROL: Control = Control {
    epoch: AtomicU32::new(0),
    n_sites: AtomicU32::new(0),
    n_workers: AtomicU32::new(0),
    workers_done: AtomicU32::new(0),
    shutdown: AtomicU32::new(0),
    sites: UnsafeCell::new([SiteJob {
        ty: 0,
        n_tokens: 0,
        n_in: 0,
        n_out: 0,
        n_blocks: 0,
        align: 1,
        w_ptr: 0,
        w_len: 0,
        q_ptr: 0,
        q_len: 0,
        y_ptr: 0,
        y_len: 0,
    }; MAX_SITES]),
};

static WORKERS: AtomicU32 = AtomicU32::new(1);

pub(crate) fn set_workers(n: u32) {
    let n = n.max(1);
    WORKERS.store(n, Ordering::Release);
}

pub(crate) fn worker_count() -> u32 {
    WORKERS.load(Ordering::Acquire)
}

pub(crate) fn pool_live() -> bool {
    worker_count() > 1
}

fn wait_i32(atom: &AtomicU32, expected: u32) {
    let p = atom as *const AtomicU32 as *mut i32;
    unsafe {
        core::arch::wasm32::memory_atomic_wait32(p, expected as i32, -1);
    }
}

fn notify_i32(atom: &AtomicU32, count: u32) {
    let p = atom as *const AtomicU32 as *mut i32;
    unsafe {
        core::arch::wasm32::memory_atomic_notify(p, count);
    }
}

pub(crate) fn worker_enter(id: u32) {
    let mut last = 0u32;
    loop {
        if CONTROL.shutdown.load(Ordering::Acquire) != 0 {
            return;
        }
        let epoch = CONTROL.epoch.load(Ordering::Acquire);
        if epoch == last {
            wait_i32(&CONTROL.epoch, epoch);
            continue;
        }
        last = epoch;
        if CONTROL.shutdown.load(Ordering::Acquire) != 0 {
            return;
        }
        let n_sites = CONTROL.n_sites.load(Ordering::Acquire) as usize;
        let n_workers = CONTROL.n_workers.load(Ordering::Acquire) as usize;
        let wid = id as usize;
        // SAFETY: coordinator wrote sites before Release on epoch; we
        // Acquire'd epoch. Pointers stay valid until workers_done join.
        let sites = unsafe { &*CONTROL.sites.get() };
        for i in 0..n_sites.min(MAX_SITES) {
            run_site(&sites[i], wid, n_workers);
        }
        let prev = CONTROL.workers_done.fetch_add(1, Ordering::Release);
        if prev + 1 >= n_workers as u32 {
            notify_i32(&CONTROL.workers_done, 1);
        }
    }
}

fn run_site(job: &SiteJob, worker: usize, n_workers: usize) {
    let n_out = job.n_out as usize;
    let align = job.align.max(1) as usize;
    let (col_start, col_end) = column_range(n_out, worker, n_workers, align);
    if col_start >= col_end {
        return;
    }
    let n_tokens = job.n_tokens as usize;
    let n_in = job.n_in as usize;
    let n_blocks = job.n_blocks as usize;
    // SAFETY: coordinator owns these allocations for the duration of the join.
    // Workers only read w/q and write disjoint y columns (see issue #44 plan).
    let w = unsafe { std::slice::from_raw_parts(job.w_ptr as *const u8, job.w_len as usize) };
    let qrows = unsafe {
        std::slice::from_raw_parts(job.q_ptr as *const BlockQ8K, job.q_len as usize)
    };
    let y = unsafe { std::slice::from_raw_parts_mut(job.y_ptr as *mut f32, job.y_len as usize) };
    let ty = match job.ty {
        TY_Q4K8 => TensorType::Q4K,
        TY_Q4K => TensorType::Q4K,
        TY_Q5K => TensorType::Q5K,
        TY_Q6K => TensorType::Q6K,
        _ => return,
    };
    matmul_ggml_cols(
        ty,
        job.ty == TY_Q4K8,
        w,
        qrows,
        n_tokens,
        n_in,
        n_out,
        n_blocks,
        y,
        col_start,
        col_end,
    );
}

fn site_from(w: &QuantMat, qrows: &[BlockQ8K], n_tokens: usize, y: &mut [f32]) -> Option<SiteJob> {
    let n_blocks = w.n_in / crate::qmatmul::QK_K;
    if n_blocks == 0 || qrows.len() != n_tokens * n_blocks {
        return None;
    }
    let (ty, align, w_ptr, w_len) = if crate::qmatmul::repack_live(w) {
        let packed = w.q4k_8x8.as_ref()?;
        (
            TY_Q4K8,
            8u32,
            packed.as_ptr() as u32,
            packed.len() as u32,
        )
    } else {
        let tag = match w.ty {
            TensorType::Q4K => TY_Q4K,
            TensorType::Q5K => TY_Q5K,
            TensorType::Q6K => TY_Q6K,
            _ => return None,
        };
        (tag, 1u32, w.bytes.as_ptr() as u32, w.bytes.len() as u32)
    };
    Some(SiteJob {
        ty,
        n_tokens: n_tokens as u32,
        n_in: w.n_in as u32,
        n_out: w.n_out as u32,
        n_blocks: n_blocks as u32,
        align,
        w_ptr,
        w_len,
        q_ptr: qrows.as_ptr() as u32,
        q_len: qrows.len() as u32,
        y_ptr: y.as_mut_ptr() as u32,
        y_len: y.len() as u32,
    })
}

fn dispatch(jobs: &[SiteJob]) -> bool {
    let n_workers = worker_count();
    if n_workers <= 1 || jobs.is_empty() || jobs.len() > MAX_SITES {
        return false;
    }
    CONTROL.n_sites.store(jobs.len() as u32, Ordering::Relaxed);
    CONTROL.n_workers.store(n_workers, Ordering::Relaxed);
    // SAFETY: workers are parked (waiting on epoch or not yet started).
    // Sites become visible after the Release fetch_add below.
    {
        let sites = unsafe { &mut *CONTROL.sites.get() };
        for (i, job) in jobs.iter().enumerate() {
            sites[i] = *job;
        }
    }
    CONTROL.workers_done.store(0, Ordering::Relaxed);
    let _ = CONTROL.epoch.fetch_add(1, Ordering::Release);
    notify_i32(&CONTROL.epoch, u32::MAX);
    loop {
        let d = CONTROL.workers_done.load(Ordering::Acquire);
        if d >= n_workers {
            break;
        }
        wait_i32(&CONTROL.workers_done, d);
    }
    true
}

/// One site. Returns false if the pool is idle (caller runs the serial gemm).
pub(crate) fn dispatch_one(
    w: &QuantMat,
    qrows: &[BlockQ8K],
    n_tokens: usize,
    y: &mut [f32],
) -> bool {
    if !pool_live() {
        return false;
    }
    let Some(job) = site_from(w, qrows, n_tokens, y) else {
        return false;
    };
    dispatch(&[job])
}

/// Two sites, one join (FFN up + gate). Same Q8_K tile, disjoint outputs.
pub(crate) fn dispatch_pair(
    w_a: &QuantMat,
    w_b: &QuantMat,
    qrows: &[BlockQ8K],
    n_tokens: usize,
    y_a: &mut [f32],
    y_b: &mut [f32],
) -> bool {
    if !pool_live() {
        return false;
    }
    let Some(a) = site_from(w_a, qrows, n_tokens, y_a) else {
        return false;
    };
    let Some(b) = site_from(w_b, qrows, n_tokens, y_b) else {
        return false;
    };
    dispatch(&[a, b])
}
