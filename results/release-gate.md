- Release gate vs bloomer: PASS
- Release gate vs bartowski: PASS

Rule: candidate KL lower on all 4 sets with the paired 95% CI of cand - comparator below 0, and cand bytes <= comparator bytes; per comparator. Intervals are pointwise.

| set | ctx | chunks | file | bytes | mean KL | p99 | top-1 % | cand - file [95% CI] |
|---|---|---|---|---|---|---|---|---|
| zh | 512 | 128 | cand | 3,116,281,408 | 0.3280 | 2.838 | 75.03 |  |
| zh | 512 | 128 | bloomer | 3,129,076,064 | 0.4128 | 2.946 | 71.72 | -0.0848 [-0.0942, -0.0756] |
| zh | 512 | 128 | bartowski | 3,235,879,360 | 0.4128 | 3.287 | 72.16 | -0.0848 [-0.0953, -0.0745] |
| zh | 512 | 128 | Q8 | 8,408,187,872 | 0.0054 | 0.058 | 96.72 |  |
| code | 512 | 128 | cand | 3,116,281,408 | 0.1464 | 1.795 | 89.00 |  |
| code | 512 | 128 | bloomer | 3,129,076,064 | 0.2499 | 2.725 | 85.66 | -0.1035 [-0.1134, -0.0940] |
| code | 512 | 128 | bartowski | 3,235,879,360 | 0.2165 | 2.463 | 86.86 | -0.0701 [-0.0790, -0.0618] |
| code | 512 | 128 | Q8 | 8,408,187,872 | 0.0025 | 0.036 | 98.66 |  |
| tool | 512 | 128 | cand | 3,116,281,408 | 0.4594 | 7.627 | 85.52 |  |
| tool | 512 | 128 | bloomer | 3,129,076,064 | 0.5830 | 8.555 | 82.76 | -0.1236 [-0.1415, -0.1059] |
| tool | 512 | 128 | bartowski | 3,235,879,360 | 0.6140 | 9.884 | 83.75 | -0.1545 [-0.1793, -0.1297] |
| tool | 512 | 128 | Q8 | 8,408,187,872 | 0.0296 | 0.569 | 96.81 |  |
| pg19 | 4096 | 32 | cand | 3,116,281,408 | 0.1232 | 0.833 | 80.56 |  |
| pg19 | 4096 | 32 | bloomer | 3,129,076,064 | 0.2286 | 1.335 | 75.21 | -0.1054 [-0.1158, -0.0955] |
| pg19 | 4096 | 32 | bartowski | 3,235,879,360 | 0.2514 | 1.639 | 74.51 | -0.1282 [-0.1465, -0.1100] |
| pg19 | 4096 | 32 | Q8 | 8,408,187,872 | 0.0013 | 0.012 | 98.12 |  |

- cand: `m2r.gguf`, 3,116,281,408 B, sha256 `1282ce370f46c09303e4dbb0d0074c7570d02645e66607603a1f5a5b84a79bd7`
- bloomer: `Ling-3.0-tiny-IQ3_XXS.gguf`, 3,129,076,064 B, sha256 `ef1c986bf0b5797c5750080985c5a9bc2013ab71430690b8e3c501cfd75d66ce`
- bartowski: `Ling-3.0-tiny-Q2_K_L.gguf`, 3,235,879,360 B, sha256 `95742a235df46b80e1ed08ee4a3c4facf76161fbe4ce7a5b403600c17d0763cb`
- Q8: `ling-3.0-tiny-Q8_0.gguf`, 8,408,187,872 B, sha256 `8acdae799ae577bb300a7b937a2cf0e3e085e1c01ccf34ad162131550a7f091c`
- PG19 cand - bloomer, book-stratified 95% interval (4 books x 8 chunks, sensitivity check): [-0.1266, -0.0856]; registered chunk interval [-0.1158, -0.0955]
- PG19 cand - bartowski, book-stratified 95% interval (4 books x 8 chunks, sensitivity check): [-0.1626, -0.0968]; registered chunk interval [-0.1465, -0.1100]
