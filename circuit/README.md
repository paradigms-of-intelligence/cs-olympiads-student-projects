# Project Documentation

### 16_types gate network:

**Binary file** size table:

| Use | Bit size |
| -------- | ------- |
| `node_count` | `32` |
| `struct Node` | `128` |
| $\dots$ | $\dots$ |
| `struct Node` | `128` |

**Node** size table:

| Use | Bit size |
| -------- | ------- |
| `type` | `32` |
| `id` | `32` |
| `link_a` | `32` |
| `link_b` | `32` |

### and_not gate network:
 :warning: The network is 1-based due to negative connections
**Binary file** size table:

| Use | Bit size |
| -------- | ------- |
| `node_count` | `32` |
| `struct Node` | `96` |
| $\dots$ | $\dots$ |
| `struct Node` | `96` |
| `output0` | `32` |
| $\dots$ | $\dots$ |
| `output9` | `32` |

**Node** size table:

| Use | Bit size |
| -------- | ------- |
| `id` | `32` |
| `link_a` | `32` |
| `link_b` | `32` |

 :warning: Links that point to $2^{31} - 1$ are to be considered always `true`. Similarly $2^{31} - 2$ links are to be considered always `false`.
 Negative links represent not nodes.

 The nodes in the network are topolgically sorted.