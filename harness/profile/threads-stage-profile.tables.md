# #56 product-path stage tables (threads + relaxed profile wasm)

Host: Intel(R) Xeon(R) Processor, Node v22.14.0, 4 CPUs.
W=1 is the threads artifact with `pool_live()=false` (serial attention).
W=4 is head-split attention (phase A2). RoPE / LN / SwiGLU stay on the coordinator.
Under W=4, qk+softmax+V-mix wall is recorded as `attn_qk` (sub-stages fused in the join).

| workers | case | n | total ms | matmul % | attn % | attn ms | RoPE ms | LN ms | SwiGLU ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | short-hello-document | 7 | 107.56 | 98.5% | 0.5% | 0.50 | 0.195 | 0.221 | 0.459 |
| 1 | long-repeated | 502 | 8823.17 | 83.0% | 15.9% | 1400.20 | 16.546 | 22.575 | 37.880 |
| 4 | short-hello-document | 7 | 41.45 | 95.9% | 1.2% | 0.51 | 0.228 | 0.229 | 0.457 |
| 4 | long-repeated | 502 | 2708.01 | 70.9% | 26.6% | 720.12 | 10.839 | 17.980 | 34.102 |

