# CVM: Compiler Backend with a Custom Stack VM

Pipeline:  `IR text -> parser -> IR -> basic blocks / CFG -> code generator -> bytecode -> VM`

```
 .ir source ──> [ IR parser ] ──> three-address IR ──> [ CFG builder ]
                                        │
                                        ▼
                               [ code generator ] ──> bytecode (.bc, 9 bytes/instr)
                                                            │
                                                            ▼
                                              [ VM: fetch-decode-execute ]
                                              operand stack + call frames
```

## Run
    python main.py examples/sum.ir --demo      # all stages
    python main.py examples/fact.ir --trace    # step-by-step VM
    python main.py examples/fact.ir --no-run -o fact.bc && python main.py fact.bc
    python tests.py                            # 17 tests

## Instruction set (24 opcodes)
| Opcode | Operands | Effect |
|---|---|---|
| PUSH | n | push constant |
| POP | | discard top |
| DUP | | duplicate top |
| LOAD / STORE | slot | push local / pop into local |
| ADD SUB MUL DIV MOD | | pop b, pop a, push a op b (DIV/MOD trap on 0) |
| LT LE GT GE EQ NE | | pop b, pop a, push 1 or 0 |
| JMP | addr | jump |
| JZ / JNZ | addr | pop; jump if zero / non-zero |
| CALL | addr, nargs | pop args into a new frame, jump |
| ENTER | nlocals | size the frame |
| RET | | pop result, drop frame, push result to caller |
| PRINT | | pop and print |
| HALT | | stop |

## Bytecode file
`"CVM1"` magic, uint32 instruction count, then fixed 9-byte instructions
(uint8 opcode, int32 a, int32 b). Jump and call targets are instruction indices.

## Code generation strategy
* Each function gets a frame; parameters occupy slots 0..n-1, other variables get slots on first use.
* Each IR statement expands to push / operate / store.
* Labels and function entries are resolved by back-patching after emission.
* Program prologue is `CALL main; HALT`.

## VM safety traps
Stack underflow (per frame), division by zero, invalid slot, invalid jump,
call-stack overflow (depth 1000), step limit (infinite loops), corrupt bytecode file.

## Planned (remaining work)
Constant folding, dead-code elimination, peephole optimiser, liveness analysis,
register allocation / register VM variant, benchmarks, optional front end.
