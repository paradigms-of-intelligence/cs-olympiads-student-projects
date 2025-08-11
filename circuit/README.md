# Project Documentation


### 16_types gate network:

**Binary file** size table:
| Use | Bit size |
| -------- | ------- |
| `node_count` | `32` |
| `struct Node` | `100` |
| $\dots$ | $\dots$ |
| `struct Node` | `100` |

**Node** size table:
| Use | Bit size |
| -------- | ------- |
| `type` | `4` |
| `id` | `32` |
| `link_a_not` | `1` |
| `link_a` | `31` |
| `link_b_not` | `31` |
| `link_b` | `31` |

### and_not gate network:
 :warning: The network is 1-based due to negative connections
**Binary file** size table:
| Use | Bit size |
| -------- | ------- |
| `node_count` | `32` |
| `struct Node` | `96` |
| $\dots$ | $\dots$ |
| `struct Node` | `96` |

**Node** size table:
| Use | Bit size |
| -------- | ------- |
| `id` | `32` |
| `link_a` | `32` |
| `link_b` | `32` |

 :warning: Links that point to $2^{31} - 1$ are to be considered always `true`. Similarly $2^{31} - 2$ links are to be considered always `false`.
 Negative links represent not nodes.