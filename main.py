import argparse
import sys

from codegen import CodegenError, generate
from ir import IRError, format_cfg, format_ir, parse
from isa import decode, disassemble, encode
from vm import VM, VMError


def banner(title):
    print(f"\n=== {title} " + "=" * max(3, 60 - len(title)))


def main():
    ap = argparse.ArgumentParser(description="CVM compiler backend + virtual machine")
    ap.add_argument("file", help=".ir source or .bc bytecode file")
    ap.add_argument("--demo", action="store_true", help="show IR, CFG and bytecode, then run")
    ap.add_argument("--ir", action="store_true", help="print the parsed IR")
    ap.add_argument("--cfg", action="store_true", help="print basic blocks and CFG edges")
    ap.add_argument("--dis", action="store_true", help="print bytecode disassembly")
    ap.add_argument("--trace", action="store_true", help="trace every VM instruction")
    ap.add_argument("-o", "--emit", metavar="OUT.bc", help="write binary bytecode to a file")
    ap.add_argument("--no-run", action="store_true", help="compile only")
    args = ap.parse_args()

    if args.demo:
        args.ir = args.cfg = args.dis = True

    try:
        if args.file.endswith(".bc"):
            with open(args.file, "rb") as f:
                code = decode(f.read())
            funcs = labels = None
            print(f"loaded {len(code)} instructions from {args.file}")
        else:
            with open(args.file) as f:
                functions = parse(f.read())
            if args.ir:
                banner("1. INTERMEDIATE REPRESENTATION (three-address code)")
                print(format_ir(functions))
            if args.cfg:
                banner("2. BASIC BLOCKS AND CONTROL FLOW GRAPH")
                print(format_cfg(functions))
            program = generate(functions)
            code, funcs, labels = program.code, program.funcs, program.labels

        if args.dis:
            banner(f"3. BYTECODE ({len(code)} instructions)")
            print(disassemble(code, funcs, labels))

        if args.emit:
            data = encode(code)
            with open(args.emit, "wb") as f:
                f.write(data)
            print(f"\nwrote {len(data)} bytes to {args.emit}")

        if args.no_run:
            return 0

        banner("4. VM EXECUTION" if (args.demo or args.dis) else "OUTPUT")
        vm = VM(code, trace=args.trace)
        vm.run()
        print(f"\n[vm] halted normally after {vm.steps} instructions")
        return 0

    except (IRError, CodegenError) as e:
        print(f"compile error: {e}", file=sys.stderr)
    except VMError as e:
        print(f"runtime error: {e}", file=sys.stderr)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
