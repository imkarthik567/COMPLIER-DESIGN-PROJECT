"""Run with:  python tests.py"""
import unittest

from codegen import CodegenError, generate
from ir import IRError, BinOp, Print, parse
from isa import Instr, Op, decode, encode
from vm import VM, VMError
from optimizer import optimize, analyze_liveness
from python_frontend import parse_python, PythonSubsetError


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
        blob = encode(code)
        self.assertEqual(blob[:4], b"CVM\x01")
        self.assertEqual(decode(blob), code)

    def test_legacy_bytecode_is_still_readable(self):
        import struct
        code = generate(parse("func main():\n print 9\n")).code
        legacy = b"CVM1" + struct.pack("<I", len(code)) + b"".join(
            struct.pack("<Bii", int(ins.op), ins.a, ins.b) for ins in code)
        self.assertEqual(decode(legacy), code)

    def test_unknown_bytecode_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported bytecode version"):
            decode(b"CVM\x02" + b"\x00\x00\x00\x00")

    def test_bad_bytecode_file(self):
        with self.assertRaises(ValueError):
            decode(b"nope")

    def test_python_frontend_range_loop(self):
        src = "total = 0\nfor i in range(1, 11):\n total += i\nprint(total)\n"
        self.assertEqual(VM(generate(parse_python(src)).code, out=lambda _x: None).run(), [55])

    def test_python_frontend_recursion(self):
        src = "def fact(n):\n if n <= 1:\n  return 1\n return n * fact(n - 1)\nprint(fact(5))\n"
        self.assertEqual(VM(generate(parse_python(src)).code, out=lambda _x: None).run(), [120])

    def test_python_frontend_main_guard(self):
        src = '"""Module docstring."""\ndef main():\n print(9)\nif __name__ == "__main__":\n main()\n'
        self.assertEqual(VM(generate(parse_python(src)).code, out=lambda _x: None).run(), [9])

    def test_python_floor_division_and_modulo_match_python(self):
        src = "print(-7 // 2)\nprint(-7 % 2)\n"
        self.assertEqual(VM(generate(parse_python(src)).code, out=lambda _x: None).run(), [-4, 1])
        self.assertEqual(VM(generate(optimize(parse_python(src))).code, out=lambda _x: None).run(), [-4, 1])

    def test_python_large_computed_integer_survives_optimization(self):
        src = "result = 50000 * 50000\nprint(result)\n"
        functions = parse_python(src)
        self.assertEqual(VM(generate(functions).code, out=lambda _x: None).run(), [2_500_000_000])
        self.assertEqual(VM(generate(optimize(functions)).code, out=lambda _x: None).run(), [2_500_000_000])

    def test_python_signed_32_bit_minimum_literal(self):
        src = "print(-2147483648)\n"
        self.assertEqual(VM(generate(parse_python(src)).code, out=lambda _x: None).run(), [-2_147_483_648])

    def test_python_frontend_rejects_unsupported_syntax(self):
        with self.assertRaises(PythonSubsetError):
            parse_python("items = [1, 2, 3]\n")

    def test_python_booleans_and_fixed_tuple_indexing(self):
        src = "enabled = True\npoint = (7, -3)\nif enabled:\n print(point[0])\n print(point[-1])\n"
        expected = VM(generate(parse_python(src)).code, out=lambda _x: None).run()
        optimized = VM(generate(optimize(parse_python(src))).code, out=lambda _x: None).run()
        self.assertEqual(expected, [7, -3])
        self.assertEqual(optimized, expected)

    def test_optimization_preserves_value_copies_after_reassignment(self):
        for src in ("a = 1\nb = a\na = 2\nprint(b)\n",
                    "a = 1\nitems = (a, 9)\na = 2\nprint(items[0])\n"):
            plain = VM(generate(parse_python(src)).code, out=lambda _x: None).run()
            optimized = VM(generate(optimize(parse_python(src))).code, out=lambda _x: None).run()
            self.assertEqual(plain, [1])
            self.assertEqual(optimized, plain)

    def test_tuple_index_must_be_static_and_in_range(self):
        for src in ("items = (1, 2)\ni = 0\nprint(items[i])\n",
                    "items = (1, 2)\nprint(items[2])\n"):
            with self.subTest(src=src), self.assertRaises(PythonSubsetError):
                parse_python(src)

    def test_python_frontend_rejects_read_before_assignment(self):
        with self.assertRaises(PythonSubsetError):
            parse_python("print(missing)\n")

    def test_python_frontend_requires_integer_function_returns(self):
        with self.assertRaises(PythonSubsetError):
            parse_python("def incomplete(x):\n if x:\n  return 1\n")

    def test_generated_temporaries_do_not_collide_with_user_variables(self):
        src = "__cvm_tmp_0 = 40\nanswer = __cvm_tmp_0 + 2\nprint(answer)\n"
        self.assertEqual(VM(generate(parse_python(src)).code, out=lambda _x: None).run(), [42])

    def test_optimized_python_preserves_output_and_reduces_steps(self):
        src = "unused = 17 * 23\nanswer = 6 * 7\ncopy = answer\nresult = copy + 0\nprint(result)\n"
        plain = VM(generate(parse_python(src)).code, out=lambda _x: None)
        optimized = VM(generate(optimize(parse_python(src))).code, out=lambda _x: None)
        self.assertEqual(plain.run(), optimized.run())
        self.assertLess(optimized.steps, plain.steps)

    def test_cfg_optimizer_propagates_constants_across_branch_join(self):
        src = ("func main():\n flag = 1\n ifz flag goto other\n value = 7\n"
               " goto join\n other:\n value = 7\n join:\n print value\n return\n")
        functions = optimize(parse(src))
        printed = next(ins for ins in functions[0].body if isinstance(ins, Print))
        self.assertEqual(printed.arg, 7)
        self.assertEqual(VM(generate(functions).code, out=lambda _x: None).run(), [7])

    def test_branch_join_demo_example_runs_and_optimizes(self):
        functions = optimize(parse(read("examples/branch_join.ir")))
        printed = next(ins for ins in functions[0].body if isinstance(ins, Print))
        self.assertEqual(printed.arg, 7)
        self.assertEqual(VM(generate(functions).code, out=lambda _x: None).run(), [7])

    def test_unsupported_demo_example_reports_subset_error(self):
        with self.assertRaises(PythonSubsetError):
            parse_python(read("examples/unsupported_python.py"))

    def test_cfg_optimizer_propagates_loop_invariant_constant(self):
        functions = optimize(parse(read("examples/sum.ir")))
        comparison = next(ins for ins in functions[0].body
                          if isinstance(ins, BinOp) and ins.op == ">")
        self.assertEqual(comparison.b, 10)
        self.assertEqual(VM(generate(functions).code, out=lambda _x: None).run(), [55])

    def test_parallel_function_optimization_preserves_program(self):
        functions = parse_python(read("examples/python_factorial.py"))
        serial = generate(optimize(functions)).code
        parallel = generate(optimize(functions, parallel=True)).code
        self.assertEqual(VM(serial, out=lambda _x: None).run(), VM(parallel, out=lambda _x: None).run())

    def test_parallel_sum_executes_worker_in_process_pool(self):
        src = ("def square(x):\n if x < 0:\n  return (0 - x) * (0 - x)\n"
               " else:\n  return x * x\nresult = parallel_sum(square, -50, 50)\nprint(result)\n")
        program = generate(parse_python(src))
        vm = VM(decode(encode(program.code)), out=lambda _x: None)
        self.assertEqual(vm.run(), [83350])
        self.assertEqual(vm.parallel_tasks, 100)
        self.assertGreaterEqual(vm.parallel_workers, 1)
        self.assertGreater(vm.parallel_worker_steps, 0)
        sequential = "total = 0\nfor x in range(-50, 50):\n total += x * x\nprint(total)\n"
        self.assertEqual(vm.output, VM(generate(parse_python(sequential)).code, out=lambda _x: None).run())

    def test_parallel_min_max_reductions(self):
        src = ("def shifted_square(value):\n return value * value - 10\n"
               "print(parallel_min(shifted_square, -5, 6))\n"
               "print(parallel_max(shifted_square, -5, 6))\n")
        for functions in (parse_python(src), optimize(parse_python(src))):
            vm = VM(generate(functions).code, out=lambda _x: None)
            self.assertEqual(vm.run(), [-10, 15])
            self.assertEqual(vm.parallel_tasks, 22)

    def test_all_parallel_reduction_operations(self):
        src = ("def positive_value(value):\n return value + 1\n"
               "print(parallel_sum(positive_value, 0, 4))\n"
               "print(parallel_min(positive_value, 0, 4))\n"
               "print(parallel_max(positive_value, 0, 4))\n"
               "print(parallel_product(positive_value, 0, 4))\n"
               "print(parallel_count(positive_value, 0, 4))\n"
               "print(parallel_any(positive_value, 0, 4))\n"
               "print(parallel_all(positive_value, 0, 4))\n"
               "print(parallel_bit_and(positive_value, 0, 4))\n"
               "print(parallel_bit_or(positive_value, 0, 4))\n"
               "print(parallel_bit_xor(positive_value, 0, 4))\n"
               "print(parallel_gcd(positive_value, 0, 4))\n"
               "print(parallel_lcm(positive_value, 0, 4))\n"
               "print(parallel_average(positive_value, 0, 4))\n"
               "print(parallel_argmin(positive_value, 0, 4))\n"
               "print(parallel_argmax(positive_value, 0, 4))\n")
        expected = [10, 1, 4, 24, 4, 1, 1, 0, 7, 4, 1, 12, 2, 0, 3]
        functions = parse_python(src)
        for version in (functions, optimize(functions)):
            code = generate(version).code
            vm = VM(decode(encode(code)), out=lambda _x: None)
            self.assertEqual(vm.run(), expected)
            self.assertEqual(vm.parallel_tasks, 60)

    def test_parallel_reduction_empty_range_identities_and_errors(self):
        identities = {
            "parallel_sum": 0, "parallel_product": 1, "parallel_count": 0,
            "parallel_any": 0, "parallel_all": 1, "parallel_bit_and": -1,
            "parallel_bit_or": 0, "parallel_bit_xor": 0, "parallel_gcd": 0,
            "parallel_lcm": 1,
        }
        for operation, expected in identities.items():
            src = f"def identity(x):\n return x\nprint({operation}(identity, 0, 0))\n"
            self.assertEqual(VM(generate(parse_python(src)).code, out=lambda _x: None).run(), [expected], operation)
        for operation in ("parallel_min", "parallel_max", "parallel_average", "parallel_argmin", "parallel_argmax"):
            src = f"def identity(x):\n return x\nprint({operation}(identity, 0, 0))\n"
            with self.subTest(operation=operation), self.assertRaisesRegex(VMError, "cannot reduce an empty range"):
                VM(generate(parse_python(src)).code, out=lambda _x: None).run()

    def test_parallel_argmin_argmax_ties_choose_lowest_input(self):
        src = ("def same(_value):\n return 7\n"
               "print(parallel_argmin(same, 2, 8))\nprint(parallel_argmax(same, 2, 8))\n")
        self.assertEqual(VM(generate(parse_python(src)).code, out=lambda _x: None).run(), [2, 2])

    def test_parallel_min_empty_range_has_clear_error(self):
        src = "def identity(x):\n return x\nprint(parallel_min(identity, 3, 3))\n"
        with self.assertRaisesRegex(VMError, "cannot reduce an empty range"):
            VM(generate(parse_python(src)).code, out=lambda _x: None).run()

    def test_cfg_liveness_finds_values_across_a_loop_edge(self):
        functions = parse("func main():\n value = 5\n loop:\n print value\n goto loop\n")
        facts = analyze_liveness(functions[0])
        self.assertTrue(any("value" in live_in for live_in, _live_out in facts.values()))

    def test_optimization_preserves_all_demonstration_outputs(self):
        for path in ("examples/sum.ir", "examples/fact.ir", "examples/primes.ir",
                     "examples/python_sum.py", "examples/python_factorial.py",
                     "examples/python_optimization.py"):
            source = read(path)
            functions = parse_python(source) if path.endswith(".py") else parse(source)
            before = VM(generate(functions).code, out=lambda _x: None).run()
            after = VM(generate(optimize(functions)).code, out=lambda _x: None).run()
            self.assertEqual(before, after, path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
