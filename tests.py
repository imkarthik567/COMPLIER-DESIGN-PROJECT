"""Run with:  python tests.py"""
import unittest

from codegen import CodegenError, generate
from ir import IRError, parse
from isa import Instr, Op, decode, encode
from vm import VM, VMError


def read(path):
    with open(path) as f:
        return f.read()


def run(src):
    out = []
    VM(generate(parse(src)).code, out=out.append).run()
    return out


class BackendTests(unittest.TestCase):
    def test_precedence_expression(self):
        self.assertEqual(run("func main():\n t=3*4\n x=2+t\n print x\n"), [14])

    def test_loop(self):
        src = read("examples/sum.ir")
        self.assertEqual(run(src), [55])

    def test_recursion(self):
        self.assertEqual(run(read("examples/fact.ir")), [120, 3628800])

    def test_nested_calls(self):
        self.assertEqual(run(read("examples/primes.ir")), [10])

    def test_truncating_division(self):
        out = run("func main():\n a=-7/2\n b=-7%2\n print a\n print b\n")
        self.assertEqual(out, [-3, -1])

    def test_discarded_call_result(self):
        src = "func f(a):\n return a\nfunc main():\n call f(3)\n print 1\n"
        self.assertEqual(run(src), [1])

    def test_division_by_zero_traps(self):
        with self.assertRaises(VMError):
            run("func main():\n a=1/0\n")

    def test_stack_underflow_traps(self):
        with self.assertRaises(VMError):
            VM([Instr(Op.ADD), Instr(Op.HALT)]).run()

    def test_call_cannot_pop_callers_operands(self):
        code = [Instr(Op.PUSH, 1), Instr(Op.CALL, 3, 0), Instr(Op.HALT),
                Instr(Op.ENTER, 0), Instr(Op.ADD), Instr(Op.RET)]
        with self.assertRaises(VMError):
            VM(code).run()

    def test_infinite_loop_is_caught(self):
        code = generate(parse("func main():\nl:\n goto l\n")).code
        with self.assertRaises(VMError):
            VM(code, max_steps=1000).run()

    def test_infinite_recursion_is_caught(self):
        src = "func f():\n x = call f()\n return x\nfunc main():\n call f()\n"
        with self.assertRaises(VMError):
            VM(generate(parse(src)).code).run()

    def test_undefined_label(self):
        with self.assertRaises(CodegenError):
            generate(parse("func main():\n goto nowhere\n"))

    def test_wrong_argument_count(self):
        with self.assertRaises(CodegenError):
            generate(parse("func f(a):\n return a\nfunc main():\n x = call f(1, 2)\n"))

    def test_missing_main(self):
        with self.assertRaises(CodegenError):
            generate(parse("func f():\n return 1\n"))

    def test_syntax_error_reports_line(self):
        with self.assertRaises(IRError) as cm:
            parse("func main():\n x = = 3\n")
        self.assertIn("line 2", str(cm.exception))

    def test_binary_roundtrip(self):
        code = generate(parse(read("examples/fact.ir"))).code
        self.assertEqual(decode(encode(code)), code)

    def test_bad_bytecode_file(self):
        with self.assertRaises(ValueError):
            decode(b"nope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
