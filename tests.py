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


from optimizer import BasicBlock, OptimizationStats, build_cfg, optimize_program


def run_opt(src):
    out = []
    opt_funcs = optimize_program(parse(src))
    VM(generate(opt_funcs).code, out=out.append).run()
    return out


class OptimizerTests(unittest.TestCase):
    def test_constant_folding_arithmetic(self):
        src = (
            "func main():\n"
            " a = 10 + 20\n"
            " b = 50 - 15\n"
            " c = 6 * 7\n"
            " d = -7 / 2\n"
            " e = -7 % 2\n"
            " print a\n print b\n print c\n print d\n print e\n"
        )
        self.assertEqual(run_opt(src), [30, 35, 42, -3, -1])

    def test_constant_folding_relational(self):
        src = (
            "func main():\n"
            " a = 3 < 5\n"
            " b = 5 <= 5\n"
            " c = 10 > 20\n"
            " d = 10 >= 10\n"
            " e = 42 == 42\n"
            " f = 42 != 42\n"
            " print a\n print b\n print c\n print d\n print e\n print f\n"
        )
        self.assertEqual(run_opt(src), [1, 1, 0, 1, 1, 0])

    def test_algebraic_simplification(self):
        src = (
            "func main():\n"
            " x = 42\n"
            " a = x + 0\n"
            " b = 0 + x\n"
            " c = x - 0\n"
            " d = x - x\n"
            " e = x * 1\n"
            " f = x * 0\n"
            " g = x / 1\n"
            " h = x % 1\n"
            " i = x == x\n"
            " j = x != x\n"
            " print a\n print b\n print c\n print d\n print e\n"
            " print f\n print g\n print h\n print i\n print j\n"
        )
        self.assertEqual(run_opt(src), [42, 42, 42, 0, 42, 0, 42, 0, 1, 0])

    def test_copy_propagation(self):
        src = (
            "func main():\n"
            " x = 10\n"
            " y = x\n"
            " z = y\n"
            " res = z + 5\n"
            " print res\n"
        )
        self.assertEqual(run_opt(src), [15])

    def test_dead_code_elimination(self):
        src = (
            "func main():\n"
            " unused1 = 100 * 2\n"
            " unused2 = 50 + unused1\n"
            " x = 7\n"
            " print x\n"
            " return\n"
        )
        funcs = optimize_program(parse(src))
        self.assertEqual(len(funcs[0].body), 2)  # only print and return
        self.assertEqual(run_opt(src), [7])

    def test_dead_store_elimination(self):
        src = (
            "func main():\n"
            " x = 10\n"
            " x = 20\n"
            " print x\n"
        )
        self.assertEqual(run_opt(src), [20])

    def test_call_side_effect_preserved_when_dst_dead(self):
        src = (
            "func f(n):\n"
            " print n\n"
            " return 999\n"
            "func main():\n"
            " dead_res = call f(123)\n"
            " print 456\n"
        )
        self.assertEqual(run_opt(src), [123, 456])

    def test_branch_folding_and_unreachable_code(self):
        src = (
            "func main():\n"
            " c = 0\n"
            " ifz c goto target\n"
            " dead_val = 999\n"
            " print dead_val\n"
            "target:\n"
            " print 1\n"
        )
        self.assertEqual(run_opt(src), [1])

    def test_cfg_dict_structure(self):
        src = read("examples/sum.ir")
        fn = parse(src)[0]
        cfg = build_cfg(fn)
        self.assertIsInstance(cfg, dict)
        self.assertIn("main_entry", cfg)
        self.assertIn("loop", cfg)
        self.assertIn("done", cfg)
        # Check basic block attributes
        b = cfg["loop"]
        self.assertIsInstance(b, BasicBlock)
        self.assertTrue(len(b.succs) > 0)
        self.assertTrue(len(b.preds) > 0)

    def test_divzero_trap_preserved_under_opt(self):
        src = "func main():\n a = 10\n b = 0\n c = a / b\n print c\n"
        with self.assertRaises(VMError):
            run_opt(src)

    def test_all_examples_equivalence(self):
        for path, expected in [
            ("examples/arith.ir", [14]),
            ("examples/sum.ir", [55]),
            ("examples/fact.ir", [120, 3628800]),
            ("examples/primes.ir", [10]),
        ]:
            src = read(path)
            self.assertEqual(run(src), expected)
            self.assertEqual(run_opt(src), expected)


from dataflow import (
    AvailableExpressions,
    DataFlowAnalysis,
    DataFlowResults,
    LivenessAnalysis,
    ReachingDefinitions,
    pretty_print,
)


class DataFlowTests(unittest.TestCase):
    def setUp(self):
        self.loop_src = (
            "func test_loop(a, b, c):\n"
            " unused_x = 100\n"
            "LOOP:\n"
            " t1 = a + b\n"
            " t2 = t1 * c\n"
            " cond = t2 > 0\n"
            " ifnz cond goto LOOP\n"
            " return t2\n"
        )
        self.fn = parse(self.loop_src)[0]
        self.cfg = build_cfg(self.fn)

    def test_liveness_on_loop_cfg(self):
        results = LivenessAnalysis().run(self.cfg)
        self.assertIsInstance(results, DataFlowResults)

        # 1. Unused variable 'unused_x' must be dead throughout
        self.assertNotIn("unused_x", results.out_sets["test_loop_entry"])
        self.assertNotIn("unused_x", results.in_sets["LOOP"])
        self.assertNotIn("unused_x", results.out_sets["LOOP"])

        # 2. Loop invariants 'a', 'b', 'c' are used in LOOP, so they are live
        self.assertTrue({"a", "b", "c"}.issubset(results.in_sets["LOOP"]))
        self.assertTrue({"a", "b", "c"}.issubset(results.out_sets["LOOP"]))

        # 3. 't1' and 'cond' are defined and used internally within LOOP, dead across backedge
        self.assertNotIn("t1", results.out_sets["LOOP"])
        self.assertNotIn("cond", results.out_sets["LOOP"])

        # 4. 't2' is returned by exit block, so it is live at exit of LOOP
        self.assertIn("t2", results.out_sets["LOOP"])

    def test_reaching_definitions_on_loop_cfg(self):
        results = ReachingDefinitions().run(self.cfg)
        self.assertIsInstance(results, DataFlowResults)

        # 1. Definition of unused_x from entry reaches LOOP
        def_x = ("unused_x", "test_loop_entry", 0)
        self.assertIn(def_x, results.out_sets["test_loop_entry"])
        self.assertIn(def_x, results.in_sets["LOOP"])

        # 2. Definitions inside LOOP: t1 (idx 1), t2 (idx 2), cond (idx 3)
        def_t1 = ("t1", "LOOP", 1)
        def_t2 = ("t2", "LOOP", 2)
        def_cond = ("cond", "LOOP", 3)

        self.assertIn(def_t1, results.out_sets["LOOP"])
        self.assertIn(def_t2, results.out_sets["LOOP"])
        self.assertIn(def_cond, results.out_sets["LOOP"])

        # 3. Across backedge, definitions generated in LOOP reach IN[LOOP]
        self.assertIn(def_t1, results.in_sets["LOOP"])
        self.assertIn(def_t2, results.in_sets["LOOP"])
        self.assertIn(def_cond, results.in_sets["LOOP"])

        # 4. In exit block, def_x and def_t2 reach entry
        exit_label = [lbl for lbl in self.cfg if lbl not in ("test_loop_entry", "LOOP")][0]
        self.assertIn(def_t2, results.in_sets[exit_label])
        self.assertIn(def_x, results.in_sets[exit_label])

    def test_available_expressions_on_loop_cfg(self):
        results = AvailableExpressions().run(self.cfg)
        self.assertIsInstance(results, DataFlowResults)

        expr_add = ("+", "a", "b")
        expr_gt = (">", "t2", 0)

        # Not available upon entry to LOOP because entry path did not evaluate it
        self.assertNotIn(expr_add, results.in_sets["LOOP"])

        # Available upon exit of LOOP and at entrance of exit block
        self.assertIn(expr_add, results.out_sets["LOOP"])
        self.assertIn(expr_gt, results.out_sets["LOOP"])

        exit_label = [lbl for lbl in self.cfg if lbl not in ("test_loop_entry", "LOOP")][0]
        self.assertIn(expr_add, results.in_sets[exit_label])
        self.assertIn(expr_gt, results.in_sets[exit_label])

    def test_available_expressions_invariant_reaches_loop(self):
        src = (
            "func test_avail(a, b):\n"
            " t0 = a + b\n"
            "LOOP:\n"
            " t1 = a + b\n"
            " cond = t1 > 0\n"
            " ifnz cond goto LOOP\n"
            " return t1\n"
        )
        fn = parse(src)[0]
        cfg = build_cfg(fn)
        results = AvailableExpressions().run(cfg)

        # 'a + b' is computed on entry and in LOOP, with neither 'a' nor 'b' redefined
        self.assertIn(("+", "a", "b"), results.in_sets["LOOP"])

    def test_pretty_print_output(self):
        results = LivenessAnalysis().run(self.cfg)
        output = pretty_print(results, self.cfg)
        self.assertIn("Block: test_loop_entry", output)
        self.assertIn("Block: LOOP", output)
        self.assertIn("IN:", output)
        self.assertIn("OUT:", output)
        self.assertIn("GEN:", output)
        self.assertIn("KILL:", output)


from ir import Assign, BinOp, Return
from optimizer import (
    CommonSubexpressionElimination,
    ConstantFoldingAndPropagation,
    CopyPropagation,
    DeadCodeElimination,
    OptimizationPass,
    OptimizationPipeline,
)


class PipelineAndPassTests(unittest.TestCase):
    def test_pass1_dce_individually(self):
        src = (
            "func test():\n"
            " x = 10\n"
            " dead1 = 20\n"
            " dead2 = dead1 + 5\n"
            " return x\n"
        )
        cfg = build_cfg(parse(src)[0])
        p = DeadCodeElimination()
        new_cfg, stats = p.run(cfg)
        self.assertGreaterEqual(stats["instructions_removed"], 2)
        remaining_dsts = [ins.dst for ins in new_cfg["test_entry"].instrs if hasattr(ins, "dst")]
        self.assertNotIn("dead1", remaining_dsts)
        self.assertNotIn("dead2", remaining_dsts)
        self.assertIn("x", remaining_dsts)

    def test_pass2_constant_folding_and_propagation_individually(self):
        src = (
            "func test():\n"
            " t1 = 3 + 4\n"
            " x = t1\n"
            " y = x * 2\n"
            " return y\n"
        )
        cfg = build_cfg(parse(src)[0])
        p = ConstantFoldingAndPropagation()
        new_cfg, stats = p.run(cfg)
        self.assertGreaterEqual(stats["constants_folded"], 1)
        first_ins = new_cfg["test_entry"].instrs[0]
        self.assertIsInstance(first_ins, Assign)
        self.assertEqual(first_ins.src, 7)

    def test_pass3_copy_propagation_individually(self):
        src = (
            "func test(a):\n"
            " x = a\n"
            " t1 = x + 5\n"
            " return t1\n"
        )
        cfg = build_cfg(parse(src)[0])
        p = CopyPropagation()
        new_cfg, stats = p.run(cfg)
        self.assertGreaterEqual(stats["copies_propagated"], 1)
        binop = [ins for ins in new_cfg["test_entry"].instrs if isinstance(ins, BinOp)][0]
        self.assertEqual(binop.a, "a")

    def test_pass4_cse_individually(self):
        src = (
            "func test(a, b):\n"
            " t1 = a * b\n"
            " t2 = a * b\n"
            " return t2\n"
        )
        cfg = build_cfg(parse(src)[0])
        p = CommonSubexpressionElimination()
        new_cfg, stats = p.run(cfg)
        self.assertEqual(stats["expressions_eliminated"], 1)
        t2_ins = new_cfg["test_entry"].instrs[1]
        self.assertIsInstance(t2_ins, Assign)
        self.assertEqual(t2_ins.src, "t1")

    def test_pipeline_run_and_convergence(self):
        src = (
            "func test(a, b):\n"
            " t1 = 3 + 4\n"
            " x = t1\n"
            " y = x + 1\n"
            " t2 = a * b\n"
            " t3 = a * b\n"
            " dead_val = 999\n"
            " res = y + t3\n"
            " return res\n"
        )
        cfg = build_cfg(parse(src)[0])
        pipeline = OptimizationPipeline()
        opt_cfg, reports = pipeline.run_until_convergence(cfg)
        self.assertTrue(len(reports) > 0)

        # Expected optimal sequence:
        # t2 = a * b
        # res = 8 + t2
        # return res
        instrs = opt_cfg["test_entry"].instrs
        self.assertEqual(len(instrs), 3)
        self.assertIsInstance(instrs[0], BinOp)
        self.assertEqual(instrs[0].dst, "t2")
        self.assertEqual(instrs[1].dst, "res")
        self.assertEqual(instrs[1].a, 8)
        self.assertEqual(instrs[1].b, "t2")
        self.assertIsInstance(instrs[2], Return)

    def test_pipeline_report_output(self):
        src = "func main():\n x = 1 + 2\n return x\n"
        cfg = build_cfg(parse(src)[0])
        pipeline = OptimizationPipeline()
        _, reports = pipeline.run(cfg)
        self.assertEqual(len(reports), len(pipeline.passes))
        # Ensure each report has required fields
        for r in reports:
            self.assertIn("pass", r)
            self.assertIn("stats", r)
            self.assertIn("time_sec", r)


from loops import (
    LoopNestingTree,
    NaturalLoop,
    build_loop_nesting_tree,
    compute_dominators,
    dominates,
    find_natural_loops,
    find_primary_induction_var,
    strictly_dominates,
)
from optimizer import (
    LoopInvariantCodeMotion,
    LoopUnrolling,
    StrengthReduction,
    cfg_to_function,
)


class LoopOptimizationTests(unittest.TestCase):
    def test_dominators_diamond_graph(self):
        src = (
            "func test(c):\n"
            " ifz c goto ELSE\n"
            " x = 1\n"
            " goto END\n"
            "ELSE:\n"
            " x = 2\n"
            "END:\n"
            " return x\n"
        )
        cfg = build_cfg(parse(src)[0])
        idom, dom, dom_tree = compute_dominators(cfg)
        entry = "test_entry"

        # Entry dominates all reachable blocks
        for b in cfg:
            self.assertTrue(dominates(entry, b, dom))
            if b != entry:
                self.assertTrue(strictly_dominates(entry, b, dom))

        # Branch blocks do not dominate the join block
        self.assertFalse(dominates("ELSE", "END", dom))
        self.assertFalse(dominates("test_b1", "END", dom))
        # Immediate dominator of END is entry
        self.assertEqual(idom["END"], entry)

    def test_natural_loop_detection(self):
        src = (
            "func test(n):\n"
            " i = 0\n"
            "LOOP:\n"
            " i = i + 1\n"
            " c = i < n\n"
            " ifnz c goto LOOP\n"
            " return i\n"
        )
        cfg = build_cfg(parse(src)[0])
        loops = find_natural_loops(cfg)
        self.assertEqual(len(loops), 1)
        loop = loops[0]
        self.assertEqual(loop.header, "LOOP")
        self.assertEqual(loop.back_edge, ("LOOP", "LOOP"))
        self.assertIn("LOOP", loop.body)
        self.assertIn("LOOP", loop.exit_blocks)

    def test_nested_loops_and_nesting_tree(self):
        src = (
            "func test(n, m):\n"
            " i = 0\n"
            "OUTER:\n"
            " j = 0\n"
            "INNER:\n"
            " j = j + 1\n"
            " c1 = j < m\n"
            " ifnz c1 goto INNER\n"
            " i = i + 1\n"
            " c2 = i < n\n"
            " ifnz c2 goto OUTER\n"
            " return i\n"
        )
        cfg = build_cfg(parse(src)[0])
        loops = find_natural_loops(cfg)
        self.assertEqual(len(loops), 2)

        # Distinguish outer and inner loop
        inner = next(l for l in loops if l.header == "INNER")
        outer = next(l for l in loops if l.header == "OUTER")

        self.assertTrue(inner.is_nested_in(outer))
        self.assertFalse(outer.is_nested_in(inner))
        self.assertTrue(inner.body.issubset(outer.body))

        tree = build_loop_nesting_tree(loops)
        self.assertEqual(tree.nesting_depth(outer), 1)
        self.assertEqual(tree.nesting_depth(inner), 2)
        self.assertEqual([l.header for l in tree.get_outermost_loops()], ["OUTER"])
        self.assertEqual([l.header for l in tree.get_innermost_loops()], ["INNER"])

    def test_primary_induction_variable(self):
        src = (
            "func test(n):\n"
            " i = 10\n"
            "LOOP:\n"
            " i = i - 2\n"
            " cond = i > 0\n"
            " ifnz cond goto LOOP\n"
            " return i\n"
        )
        cfg = build_cfg(parse(src)[0])
        loops = find_natural_loops(cfg)
        iv = find_primary_induction_var(loops[0], cfg)
        self.assertIsNotNone(iv)
        var_name, step, block, idx, ins = iv
        self.assertEqual(var_name, "i")
        self.assertEqual(step, -2)
        self.assertEqual(block, "LOOP")

    def test_pass5_licm_individually(self):
        src = (
            "func test(a, b):\n"
            " i = 0\n"
            "LOOP:\n"
            " inv = a + b\n"
            " i = i + 1\n"
            " c = i < 10\n"
            " ifnz c goto LOOP\n"
            " return inv\n"
        )
        cfg = build_cfg(parse(src)[0])
        p = LoopInvariantCodeMotion()
        new_cfg, stats = p.run(cfg)

        self.assertEqual(stats["instructions_hoisted"], 1)
        self.assertIn("LOOP_preheader", new_cfg)
        preheader_instrs = new_cfg["LOOP_preheader"].instrs
        self.assertTrue(any(isinstance(ins, BinOp) and ins.dst == "inv" for ins in preheader_instrs))

        # 'inv = a + b' must no longer be inside LOOP
        loop_dsts = [ins.dst for ins in new_cfg["LOOP"].instrs if hasattr(ins, "dst")]
        self.assertNotIn("inv", loop_dsts)

        # Preheader must be linked properly
        self.assertIn("LOOP", new_cfg["LOOP_preheader"].succs)
        self.assertIn("LOOP_preheader", new_cfg["LOOP"].preds)

    def test_pass5_licm_chained_invariants(self):
        src = (
            "func test(a):\n"
            " i = 0\n"
            "LOOP:\n"
            " t1 = a + 5\n"
            " t2 = t1 * 2\n"
            " i = i + 1\n"
            " c = i < 10\n"
            " ifnz c goto LOOP\n"
            " return t2\n"
        )
        cfg = build_cfg(parse(src)[0])
        p = LoopInvariantCodeMotion()
        new_cfg, stats = p.run(cfg)

        self.assertEqual(stats["instructions_hoisted"], 2)
        pre_dsts = [ins.dst for ins in new_cfg["LOOP_preheader"].instrs if hasattr(ins, "dst")]
        self.assertEqual(pre_dsts, ["t1", "t2"])

    def test_pass6_strength_reduction_individually(self):
        src = (
            "func main():\n"
            " i = 0\n"
            " sum = 0\n"
            "LOOP:\n"
            " t = i * 4\n"
            " sum = sum + t\n"
            " i = i + 1\n"
            " c = i < 5\n"
            " ifnz c goto LOOP\n"
            " return sum\n"
        )
        cfg = build_cfg(parse(src)[0])
        p = StrengthReduction()
        new_cfg, stats = p.run(cfg)

        self.assertGreaterEqual(stats["operations_reduced"], 1)
        # Multiplication in loop replaced with copy
        loop_binops = [ins for ins in new_cfg["LOOP"].instrs if isinstance(ins, BinOp)]
        mults = [b for b in loop_binops if b.op == "*"]
        self.assertEqual(len(mults), 0)

        # Execution correctness verified on VM
        opt_fn = cfg_to_function(parse(src)[0], new_cfg)
        out = []
        # sum = 0*4 + 1*4 + 2*4 + 3*4 + 4*4 = 40
        src_run = src.replace("return sum", "print sum\n return sum")
        self.assertEqual(run_opt(src_run), [40])

    def test_pass7_loop_unrolling_small_trip_count(self):
        src = (
            "func main():\n"
            " i = 0\n"
            " sum = 0\n"
            "LOOP:\n"
            " sum = sum + 10\n"
            " i = i + 1\n"
            " c = i < 3\n"
            " ifnz c goto LOOP\n"
            " print sum\n"
        )
        cfg = build_cfg(parse(src)[0])
        p = LoopUnrolling()
        new_cfg, stats = p.run(cfg)

        self.assertEqual(stats["loops_unrolled"], 1)
        # Backedge should be removed from LOOP
        self.assertNotIn("LOOP", new_cfg["LOOP"].succs)

        # Execution check
        self.assertEqual(run_opt(src), [30])

    def test_pass7_loop_unrolling_call_guard(self):
        src = (
            "func dummy():\n"
            " return 1\n"
            "func main():\n"
            " i = 0\n"
            "LOOP:\n"
            " call dummy()\n"
            " i = i + 1\n"
            " c = i < 3\n"
            " ifnz c goto LOOP\n"
            " print i\n"
        )
        cfg = build_cfg(parse(src)[1])
        p = LoopUnrolling()
        new_cfg, stats = p.run(cfg)
        self.assertEqual(stats["loops_unrolled"], 0)


import struct
from verifier import (
    FuncRef,
    Type,
    VerificationError,
    VerificationResult,
    VerificationWarning,
    Verifier,
)


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.v = Verifier()

    def test_valid_bytecode_passes(self):
        for path in ["examples/arith.ir", "examples/sum.ir", "examples/fact.ir", "examples/primes.ir"]:
            src = read(path)
            prog = generate(parse(src))
            result = self.v.verify(prog.code)
            self.assertTrue(result.passed)
            self.assertEqual(len(result.errors), 0)
            self.assertGreater(result.stats["instruction_count"], 0)
            self.assertGreater(result.stats["functions_found"], 0)
            self.assertGreater(result.stats["stack_depth_max"], 0)

    def test_magic_number_check(self):
        # Good magic bytes
        valid_data = encode([Instr(Op.HALT)])
        self.assertTrue(self.v.verify(valid_data).passed)

        # Bad magic bytes
        bad_data = b"BAD1" + valid_data[4:]
        res = self.v.verify(bad_data)
        self.assertFalse(res.passed)
        self.assertTrue(any(e.code == "BAD_MAGIC" for e in res.errors))

    def test_instruction_count_mismatch(self):
        # Header declares 100 instructions, but only 1 exists
        header = b"CVM1" + struct.pack("<I", 100) + struct.pack("<Bii", int(Op.HALT), 0, 0)
        res = self.v.verify(header)
        self.assertFalse(res.passed)
        self.assertTrue(any(e.code == "COUNT_MISMATCH" for e in res.errors))

    def test_jump_to_invalid_target_fails(self):
        # JMP out of bounds (999)
        code = [
            Instr(Op.CALL, 2, 0),
            Instr(Op.HALT),
            Instr(Op.ENTER, 0),
            Instr(Op.JMP, 999),
            Instr(Op.RET),
        ]
        res = self.v.verify(code)
        self.assertFalse(res.passed)
        self.assertTrue(any(e.code == "BAD_JUMP_TARGET" for e in res.errors))

        # Negative jump target (-1)
        code[3] = Instr(Op.JMP, -1)
        res = self.v.verify(code)
        self.assertFalse(res.passed)
        self.assertTrue(any(e.code == "BAD_JUMP_TARGET" for e in res.errors))

    def test_stack_underflow_fails(self):
        # Binary ADD with 0 operands
        code = [
            Instr(Op.CALL, 2, 0),
            Instr(Op.HALT),
            Instr(Op.ENTER, 0),
            Instr(Op.ADD),
            Instr(Op.RET),
        ]
        res = self.v.verify(code)
        self.assertFalse(res.passed)
        self.assertTrue(any(e.code == "STACK_UNDERFLOW" for e in res.errors))

        # POP with 0 operands
        code[3] = Instr(Op.POP)
        res = self.v.verify(code)
        self.assertFalse(res.passed)
        self.assertTrue(any(e.code == "STACK_UNDERFLOW" for e in res.errors))

    def test_unterminated_function_fails(self):
        # Function does not end with RET
        code = [
            Instr(Op.CALL, 2, 0),
            Instr(Op.HALT),
            Instr(Op.ENTER, 1),
            Instr(Op.PUSH, 42),
            Instr(Op.STORE, 0),
        ]
        res = self.v.verify(code)
        self.assertFalse(res.passed)
        self.assertTrue(any(e.code in ("UNTERMINATED_FUNCTION", "FALL_OFF_END") for e in res.errors))

    def test_dead_code_after_jmp_produces_warning(self):
        # Unreachable code after unconditional JMP
        code = [
            Instr(Op.CALL, 2, 0),
            Instr(Op.HALT),
            Instr(Op.ENTER, 0),
            Instr(Op.JMP, 6),
            Instr(Op.PUSH, 99),  # index 4: dead
            Instr(Op.POP),       # index 5: dead
            Instr(Op.PUSH, 0),   # index 6: target
            Instr(Op.RET),       # index 7
        ]
        res = self.v.verify(code)
        self.assertTrue(res.passed)
        self.assertTrue(len(res.warnings) > 0)
        warning_indices = [w.instruction_index for w in res.warnings]
        self.assertIn(4, warning_indices)
        self.assertTrue(any(w.code == "DEAD_CODE" for w in res.warnings))

    def test_stack_depth_mismatch_at_merge_point(self):
        # One branch leaves depth 1, another branch leaves depth 2 before join
        code = [
            Instr(Op.CALL, 2, 0),
            Instr(Op.HALT),
            Instr(Op.ENTER, 0),
            Instr(Op.PUSH, 1),
            Instr(Op.JZ, 7),
            Instr(Op.PUSH, 10),  # depth becomes 1
            Instr(Op.JMP, 9),
            Instr(Op.PUSH, 20),  # depth at 7 is 0, then 1
            Instr(Op.PUSH, 30),  # depth at 8 is 2
            Instr(Op.RET),       # join point at 9: incoming 1 vs 2!
        ]
        res = self.v.verify(code)
        self.assertFalse(res.passed)
        self.assertTrue(any(e.code == "STACK_DEPTH_MISMATCH" for e in res.errors))

    def test_stack_not_empty_at_ret(self):
        # Stack depth is 2 at RET (1 value leftover)
        code = [
            Instr(Op.CALL, 2, 0),
            Instr(Op.HALT),
            Instr(Op.ENTER, 0),
            Instr(Op.PUSH, 1),
            Instr(Op.PUSH, 2),
            Instr(Op.RET),
        ]
        res = self.v.verify(code)
        self.assertFalse(res.passed)
        self.assertTrue(any(e.code == "STACK_NOT_EMPTY" for e in res.errors))

    def test_slot_index_out_of_bounds(self):
        # Frame size is 2 (slots 0..1), but STORE references slot 2
        code = [
            Instr(Op.CALL, 2, 0),
            Instr(Op.HALT),
            Instr(Op.ENTER, 2),
            Instr(Op.PUSH, 42),
            Instr(Op.STORE, 2),
            Instr(Op.PUSH, 0),
            Instr(Op.RET),
        ]
        res = self.v.verify(code)
        self.assertFalse(res.passed)
        self.assertTrue(any(e.code == "BAD_SLOT" for e in res.errors))

        # Negative slot index
        code[4] = Instr(Op.STORE, -1)
        res = self.v.verify(code)
        self.assertFalse(res.passed)
        self.assertTrue(any(e.code == "BAD_SLOT" for e in res.errors))

    def test_garbage_nan_operand(self):
        code = [
            Instr(Op.CALL, 2, 0),
            Instr(Op.HALT),
            Instr(Op.ENTER, 1),
            Instr(Op.PUSH, float("nan")),
            Instr(Op.STORE, 0),
            Instr(Op.PUSH, 0),
            Instr(Op.RET),
        ]
        res = self.v.verify(code)
        self.assertFalse(res.passed)
        self.assertTrue(any(e.code == "GARBAGE_OPERAND" for e in res.errors))

    def test_invalid_opcode(self):
        code = [
            Instr(Op.CALL, 2, 0),
            Instr(Op.HALT),
            Instr(Op.ENTER, 0),
            Instr(99),  # invalid opcode
            Instr(Op.RET),
        ]
        res = self.v.verify(code)
        self.assertFalse(res.passed)
        self.assertTrue(any(e.code == "INVALID_OPCODE" for e in res.errors))

    def test_call_target_must_be_enter(self):
        # CALL targets HALT (instruction 1) instead of an ENTER instruction
        code = [
            Instr(Op.CALL, 1, 0),
            Instr(Op.HALT),
            Instr(Op.ENTER, 0),
            Instr(Op.PUSH, 0),
            Instr(Op.RET),
        ]
        res = self.v.verify(code)
        self.assertFalse(res.passed)
        self.assertTrue(any(e.code == "INVALID_FUNCTION_ENTRY" for e in res.errors))

    def test_arithmetic_on_funcref_fails(self):
        # Arithmetic on function reference must be rejected
        code = [
            Instr(Op.CALL, 2, 0),
            Instr(Op.HALT),
            Instr(Op.ENTER, 1),
            Instr(Op.PUSH, FuncRef(2)),
            Instr(Op.STORE, 0),
            Instr(Op.LOAD, 0),
            Instr(Op.PUSH, 5),
            Instr(Op.ADD),
            Instr(Op.RET),
        ]
        res = self.v.verify(code)
        self.assertFalse(res.passed)
        self.assertTrue(any(e.code == "TYPE_MISMATCH" for e in res.errors))

    def test_max_stack_depth_limit(self):
        # Verifier with stack limit 2
        v_strict = Verifier(max_stack_depth=2)
        code = [
            Instr(Op.CALL, 2, 0),
            Instr(Op.HALT),
            Instr(Op.ENTER, 0),
            Instr(Op.PUSH, 1),
            Instr(Op.PUSH, 2),
            Instr(Op.PUSH, 3),  # depth exceeds 2
            Instr(Op.POP),
            Instr(Op.POP),
            Instr(Op.RET),
        ]
        res = v_strict.verify(code)
        self.assertFalse(res.passed)
        self.assertTrue(any(e.code == "STACK_OVERFLOW" for e in res.errors))


from regalloc import (
    InterferenceGraph,
    RegisterAllocator,
    RegAllocStats,
    SpillSlotAllocator,
    allocate_registers_for_program,
    build_interference_graph,
    coalesce_copies,
    color_graph,
    insert_spill_code,
)


def run_regalloc(src):
    p = parse(src)
    reg_maps, new_p, stats = allocate_registers_for_program(p)
    out = []
    vm = VM(generate(new_p, reg_map=reg_maps).code, out=out.append)
    vm.run()
    return out


class RegisterAllocatorTests(unittest.TestCase):
    def test_interference_graph_basic(self):
        src = """func main():
            a = 10
            b = 20
            c = a + b
            print c
            return
        """
        fn = parse(src)[0]
        cfg = build_cfg(fn)
        graph, copies, u_d = build_interference_graph(fn, cfg)
        # a and b are both live at the point of c = a + b, so they interfere
        self.assertIn("b", graph.neighbors("a"))
        self.assertIn("a", graph.neighbors("b"))
        self.assertEqual(graph.degree("a"), len(graph.neighbors("a")))

    def test_copy_non_interference(self):
        # In x = y, x and y do not interfere by virtue of this copy
        src = """func main():
            y = 5
            x = y
            print x
            return
        """
        fn = parse(src)[0]
        cfg = build_cfg(fn)
        graph, copies, u_d = build_interference_graph(fn, cfg)
        self.assertNotIn("y", graph.neighbors("x"))
        self.assertIn(("x", "y"), copies)

    def test_conservative_coalescing(self):
        src = """func main():
            y = 42
            x = y
            print x
            return
        """
        fn = parse(src)[0]
        cfg = build_cfg(fn)
        graph, copies, u_d = build_interference_graph(fn, cfg)
        alias, coalesced_count = coalesce_copies(graph, copies, K=8)
        self.assertGreaterEqual(coalesced_count, 1)
        self.assertTrue(alias.get("x") == "y" or alias.get("y") == "x")

    def test_graph_coloring_k8_no_spill(self):
        src = """func main():
            a = 1
            b = 2
            c = 3
            d = 4
            t = a + b
            u = c + d
            ans = t + u
            print ans
            return
        """
        fn = parse(src)[0]
        cfg = build_cfg(fn)
        graph, copies, u_d = build_interference_graph(fn, cfg)
        colors, spills = color_graph(graph, u_d, K=8)
        self.assertEqual(len(spills), 0)
        # Verify valid coloring (no two neighbors share color)
        for u in graph.nodes():
            for v in graph.neighbors(u):
                if u in colors and v in colors:
                    self.assertNotEqual(colors[u], colors[v])

    def test_spill_priority_heuristic(self):
        # Node with high degree and low uses/defs gets lower priority (chosen to spill first)
        g = InterferenceGraph(["frequent", "infrequent"])
        g.add_edge("frequent", "infrequent")
        u_d = {"frequent": 100, "infrequent": 1}
        # K=1 -> one must spill
        colors, spills = color_graph(g, u_d, K=1)
        self.assertIn("infrequent", spills)
        self.assertIn("frequent", colors)

    def test_spill_code_insertion(self):
        src = """func main():
            a = 1
            b = 2
            c = a + b
            print c
            return
        """
        fn = parse(src)[0]
        spill_alloc = SpillSlotAllocator(start_slot=0)
        new_fn = insert_spill_code(fn, {"a"}, spill_alloc, iteration=1)
        # Verify 'a' definition was replaced and reload was inserted before use
        has_reload = any(isinstance(ins, Assign) and "a_load" in ins.dst for ins in new_fn.body)
        has_store = any(isinstance(ins, Assign) and ins.dst == "_spill_slot_0" for ins in new_fn.body)
        self.assertTrue(has_reload)
        self.assertTrue(has_store)

    def test_forced_spills_large_live_range(self):
        # 10 live variables simultaneously -> exceeds K=8 registers, forces spilling
        src = """func main():
            a = 1
            b = 2
            c = 3
            d = 4
            e = 5
            f = 6
            g = 7
            h = 8
            i = 9
            j = 10
            t1 = a + b
            t2 = t1 + c
            t3 = t2 + d
            t4 = t3 + e
            t5 = t4 + f
            t6 = t5 + g
            t7 = t6 + h
            t8 = t7 + i
            total = t8 + j
            print total
            return
        """
        p = parse(src)
        reg_maps, new_p, stats = allocate_registers_for_program(p, K=8)
        self.assertGreater(stats.spills, 0)
        self.assertGreater(stats.assigned_to_registers, 0)
        # VM execution must still compute exact result
        out = run_regalloc(src)
        self.assertEqual(out, [55])

    def test_examples_execution_with_regalloc(self):
        # Verify exact semantic equivalence with pure stack on all examples
        for path, exp in [
            ("examples/sum.ir", [55]),
            ("examples/fact.ir", [120, 3628800]),
            ("examples/primes.ir", [10]),
        ]:
            src = read(path)
            stack_out = run(src)
            reg_out = run_regalloc(src)
            self.assertEqual(reg_out, exp)
            self.assertEqual(reg_out, stack_out)

    def test_regalloc_stats_report(self):
        stats = RegAllocStats(
            total_variables=12,
            assigned_to_registers=10,
            spills=2,
            copies_coalesced=3,
            estimated_instruction_reduction=9,
        )
        report = stats.format_report()
        self.assertIn("REGISTER ALLOCATION REPORT", report)
        self.assertIn("Total variables:             12", report)
        self.assertIn("Assigned to registers:       10", report)
        self.assertIn("Spills to memory:            2", report)
        self.assertIn("Copies coalesced:            3", report)
        self.assertIn("Estimated instrs saved:      9", report)
        self.assertEqual(stats.report(), report)

    def test_codegen_physical_register_opcodes(self):
        # Verify RLOAD, RLOAD_IMM, RSTORE are generated when using registers
        src = """func main():
            x = 10
            y = x
            print y
            return
        """
        p = parse(src)
        reg_maps, new_p, stats = allocate_registers_for_program(p)
        code = generate(new_p, reg_map=reg_maps).code
        opcodes = {ins.op for ins in code}
        # Either RLOAD_IMM was generated or RLOAD
        self.assertTrue(Op.RLOAD_IMM in opcodes or Op.RLOAD in opcodes)


from ir import Assign, BinOp, Call, Goto, IfNz, IfZ, Label, Print, Return
from scheduler import (
    BlockScheduleReport,
    DataDependencyGraph,
    DDGEdge,
    DDGNode,
    DependencyType,
    InstructionScheduler,
    SchedulingStats,
    Superblock,
    build_ddg,
    compute_block_frequency_hints,
    estimate_pipeline_cycles,
    form_superblocks,
    get_instruction_latency,
    is_barrier,
    is_call,
    is_store,
    list_schedule,
    schedule_program,
    schedule_superblock,
    verify_schedule,
)


def run_scheduled(src, use_superblocks=False):
    p = parse(src)
    reg_maps, new_p, stats = allocate_registers_for_program(p)
    sched_p, s_stats = schedule_program(new_p, reg_map=reg_maps, use_superblocks=use_superblocks)
    out = []
    vm = VM(generate(sched_p, reg_map=reg_maps).code, out=out.append)
    vm.run()
    return out


class InstructionSchedulerTests(unittest.TestCase):
    def test_ddg_raw_dependency(self):
        src = """func main():
            a = 10
            b = a + 5
            print b
            return
        """
        fn = parse(src)[0]
        cfg = build_cfg(fn)
        entry_block = cfg[f"{fn.name}_entry"]
        ddg = build_ddg(entry_block.instrs)
        # a = 10 (idx 0) -> b = a + 5 (idx 1) is a RAW dependency
        edge = ddg.edge_map.get((0, 1))
        self.assertIsNotNone(edge)
        self.assertEqual(edge.dep_type, DependencyType.RAW)
        self.assertEqual(edge.weight, ddg.latency(0))

    def test_ddg_war_dependency(self):
        # b = a + 5 reads a; then a = 20 writes a -> WAR dependency (idx 0 -> idx 1)
        instrs = [
            BinOp("b", "+", "a", 5),
            Assign("a", 20),
        ]
        ddg = build_ddg(instrs)
        edge = ddg.edge_map.get((0, 1))
        self.assertIsNotNone(edge)
        self.assertEqual(edge.dep_type, DependencyType.WAR)
        self.assertEqual(edge.weight, ddg.latency(0))

    def test_ddg_waw_dependency(self):
        # a = 10 writes a; a = 20 writes a -> WAW dependency (idx 0 -> idx 1)
        instrs = [
            Assign("a", 10),
            Assign("a", 20),
        ]
        ddg = build_ddg(instrs)
        edge = ddg.edge_map.get((0, 1))
        self.assertIsNotNone(edge)
        self.assertEqual(edge.dep_type, DependencyType.WAW)
        self.assertEqual(edge.weight, ddg.latency(0))

    def test_ddg_control_barrier_branch(self):
        # Branches are barriers; any preceding instruction must precede the branch
        instrs = [
            Assign("x", 1),
            Assign("y", 2),
            Goto("somewhere"),
        ]
        ddg = build_ddg(instrs)
        self.assertIn(2, ddg.get_successors(0))
        self.assertIn(2, ddg.get_successors(1))
        self.assertEqual(ddg.edge_map[(0, 2)].dep_type, DependencyType.CONTROL)
        self.assertEqual(ddg.edge_map[(1, 2)].dep_type, DependencyType.CONTROL)

    def test_ddg_control_barrier_call(self):
        # Calls are full barriers: nothing moves past them in either direction
        instrs = [
            Assign("x", 1),
            Call(None, "foo", []),
            Assign("y", 2),
        ]
        ddg = build_ddg(instrs)
        # x -> call (must precede call)
        self.assertIn(1, ddg.get_successors(0))
        # call -> y (must follow call)
        self.assertIn(2, ddg.get_successors(1))
        self.assertEqual(ddg.edge_map[(0, 1)].dep_type, DependencyType.CONTROL)
        self.assertEqual(ddg.edge_map[(1, 2)].dep_type, DependencyType.CONTROL)

    def test_ddg_instruction_latencies(self):
        self.assertEqual(get_instruction_latency(BinOp("c", "+", "a", "b")), 1)
        self.assertEqual(get_instruction_latency(BinOp("c", "-", "a", "b")), 1)
        self.assertEqual(get_instruction_latency(BinOp("c", "<", "a", "b")), 1)
        self.assertEqual(get_instruction_latency(BinOp("c", "*", "a", "b")), 3)
        self.assertEqual(get_instruction_latency(BinOp("c", "/", "a", "b")), 5)
        self.assertEqual(get_instruction_latency(BinOp("c", "%", "a", "b")), 5)
        self.assertEqual(get_instruction_latency(Assign("x", "mem_slot")), 2)
        self.assertEqual(get_instruction_latency(Assign("_spill_slot_0", "x")), 1)
        self.assertEqual(get_instruction_latency(Call("res", "f", [])), 10)

    def test_critical_path_linear_chain(self):
        # a = 1 (lat 1) -> b = a * 2 (lat 3) -> c = b + 3 (lat 1)
        instrs = [
            Assign("a", 1),
            BinOp("b", "*", "a", 2),
            BinOp("c", "+", "b", 3),
        ]
        ddg = build_ddg(instrs)
        cp = ddg.compute_critical_paths()
        # c is leaf: cp = 1
        self.assertEqual(cp[2], 1)
        # b -> c: cp = 3 + 1 = 4
        self.assertEqual(cp[1], 4)
        # a -> b -> c: cp = 1 + 3 + 1 = 5
        self.assertEqual(cp[0], 5)

    def test_critical_path_diamond_graph(self):
        # Node 0 forks into Node 1 (Multiply, lat 3) and Node 2 (Add, lat 1), both join at Node 3 (Add, lat 1)
        instrs = [
            Assign("a", 10),
            BinOp("m", "*", "a", 2),
            BinOp("s", "+", "a", 2),
            BinOp("res", "+", "m", "s"),
        ]
        ddg = build_ddg(instrs)
        cp = ddg.compute_critical_paths()
        # Path through Multiply is longer: 1 (a) + 3 (m) + 1 (res) = 5
        self.assertEqual(cp[3], 1)
        self.assertEqual(cp[1], 4)  # 3 + 1
        self.assertEqual(cp[2], 2)  # 1 + 1
        self.assertEqual(cp[0], 5)  # 1 + max(4, 2) = 5

    def test_list_scheduling_independent_chains_hides_latency(self):
        # Two independent multiply-add chains
        # Original: t1 = a * b; r1 = t1 + 10; t2 = c * d; r2 = t2 + 20
        # Scheduled: interleaves t2 = c * d between t1 and r1 to fill stall cycles
        instrs = [
            Assign("a", 2),
            Assign("b", 3),
            BinOp("t1", "*", "a", "b"),
            BinOp("r1", "+", "t1", 10),
            Assign("c", 4),
            Assign("d", 5),
            BinOp("t2", "*", "c", "d"),
            BinOp("r2", "+", "t2", 20),
        ]
        sched = list_schedule(instrs)
        c_orig, s_orig = estimate_pipeline_cycles(instrs)
        c_sched, s_sched = estimate_pipeline_cycles(sched)
        self.assertLess(s_sched, s_orig)
        self.assertGreater(s_orig - s_sched, 0)
        self.assertLess(c_sched, c_orig)

    def test_list_scheduling_tie_breaking(self):
        # When critical paths are equal, tie-breaker prioritizes: calls > stores > other
        ins_other = BinOp("res_other", "+", "a", "b")
        ins_store = Assign("_spill_slot_0", "a")
        ins_call = Call("res_call", "foo", [])
        # Both are ready with cp = latency
        self.assertTrue(is_call(ins_call))
        self.assertTrue(is_store(ins_store))
        self.assertFalse(is_call(ins_other))
        self.assertFalse(is_store(ins_other))

    def test_verify_schedule_detects_violations(self):
        instrs = [
            Assign("a", 1),
            BinOp("b", "+", "a", 2),
        ]
        ddg = build_ddg(instrs)
        # Valid schedule
        self.assertTrue(verify_schedule(instrs, instrs, ddg))
        # Inverted schedule violates RAW dependency: b depends on a
        inverted = [instrs[1], instrs[0]]
        with self.assertRaises(AssertionError):
            verify_schedule(instrs, inverted, ddg, scheduled_order=[1, 0])

    def test_pipeline_cycle_and_stall_estimation(self):
        # Single multiply followed by dependent add: stalls = 3 - 1 = 2 cycles
        instrs = [
            BinOp("m", "*", 2, 3),
            BinOp("res", "+", "m", 1),
        ]
        cycles, stalls = estimate_pipeline_cycles(instrs)
        self.assertEqual(stalls, 2)
        self.assertEqual(cycles, 4)

    def test_superblock_formation_and_frequency_hints(self):
        src = read("examples/sum.ir")
        fn = parse(src)[0]
        cfg = build_cfg(fn)
        freq = compute_block_frequency_hints(cfg)
        # Loop blocks must have higher execution frequency than non-loop entry
        self.assertGreater(freq.get("loop", 0), freq.get("main_entry", 0))
        sbs = form_superblocks(cfg)
        self.assertGreater(len(sbs), 0)

    def test_superblock_scheduling_execution(self):
        for path, exp in [
            ("examples/sum.ir", [55]),
            ("examples/fact.ir", [120, 3628800]),
            ("examples/primes.ir", [10]),
        ]:
            src = read(path)
            out = run_scheduled(src, use_superblocks=True)
            self.assertEqual(out, exp)

    def test_all_examples_execution_with_scheduler(self):
        for path, exp in [
            ("examples/sum.ir", [55]),
            ("examples/fact.ir", [120, 3628800]),
            ("examples/primes.ir", [10]),
        ]:
            src = read(path)
            out_pure = run(src)
            out_sched = run_scheduled(src, use_superblocks=False)
            self.assertEqual(out_sched, exp)
            self.assertEqual(out_sched, out_pure)

    def test_full_compiler_pipeline_opt_regalloc_scheduler(self):
        # Entire backend pipeline: optimizer -> regalloc -> scheduler -> codegen -> VM
        for path, exp in [
            ("examples/sum.ir", [55]),
            ("examples/fact.ir", [120, 3628800]),
            ("examples/primes.ir", [10]),
        ]:
            p = parse(read(path))
            opt_p = optimize_program(p)
            reg_map, reg_p, reg_stats = allocate_registers_for_program(opt_p)
            sched_p, sched_stats = schedule_program(reg_p)
            out = []
            vm = VM(generate(sched_p, reg_map=reg_map).code, out=out.append)
            vm.run()
            self.assertEqual(out, exp)

    def test_scheduling_stats_report_format(self):
        src = """func main():
            a = 2
            b = 3
            t1 = a * b
            r1 = t1 + 10
            c = 4
            d = 5
            t2 = c * d
            r2 = t2 + 20
            ans = r1 + r2
            print ans
            return
        """
        p = parse(src)
        sched_p, stats = schedule_program(p)
        report = stats.format_report()
        self.assertIn("INSTRUCTION SCHEDULING REPORT", report)
        self.assertIn("Total stall cycles saved:", report)
        self.assertIn("Blocks Ranked by Stall Cycles Saved:", report)
        self.assertGreater(stats.total_stalls_saved, 0)
        self.assertEqual(stats.report(), report)


# =====================================================================
# Phase 7: SIMD Loop Vectorizer Tests
# =====================================================================

import ir
import vectorizer
from vectorizer import (
    ArrayAccess,
    LoopVectorizationResult,
    VectorizationStats,
    check_vectorizability,
    gcd_dependence_test,
    vectorize_function,
    vectorize_program,
)


class VectorizerTests(unittest.TestCase):
    def test_gcd_dependence_independent_accesses(self):
        # A[2*i] vs A[2*i + 1]: gcd(2, 2) = 2 does not divide (1 - 0) = 1
        acc1 = ArrayAccess(array="a", is_store=True, offset=0, ins=None, coeff=2)
        acc2 = ArrayAccess(array="a", is_store=False, offset=1, ins=None, coeff=2)
        has_dep, dist = gcd_dependence_test(acc1, acc2)
        self.assertFalse(has_dep)
        self.assertIsNone(dist)

    def test_gcd_dependence_intra_iteration(self):
        # A[i] write and A[i] read: distance 0 -> same iteration (intra-iteration)
        acc1 = ArrayAccess(array="a", is_store=True, offset=0, ins=None, coeff=1)
        acc2 = ArrayAccess(array="a", is_store=False, offset=0, ins=None, coeff=1)
        has_dep, dist = gcd_dependence_test(acc1, acc2)
        self.assertFalse(has_dep)
        self.assertEqual(dist, 0)

    def test_gcd_dependence_flow_dependence(self):
        # A[i] write vs A[i-1] read: distance 1 -> loop-carried cross-iteration dependence
        acc1 = ArrayAccess(array="a", is_store=True, offset=0, ins=None, coeff=1)
        acc2 = ArrayAccess(array="a", is_store=False, offset=-1, ins=None, coeff=1)
        has_dep, dist = gcd_dependence_test(acc1, acc2)
        self.assertTrue(has_dep)
        self.assertEqual(dist, 1)

    def test_gcd_dependence_read_after_read(self):
        # Two reads on same array with different offsets: Read-After-Read is hazard-free
        acc1 = ArrayAccess(array="a", is_store=False, offset=0, ins=None, coeff=1)
        acc2 = ArrayAccess(array="a", is_store=False, offset=1, ins=None, coeff=1)
        has_dep, dist = gcd_dependence_test(acc1, acc2)
        self.assertFalse(has_dep)

    def test_gcd_dependence_different_arrays(self):
        # Different arrays A and B cannot interfere
        acc1 = ArrayAccess(array="a", is_store=True, offset=0, ins=None, coeff=1)
        acc2 = ArrayAccess(array="b", is_store=False, offset=0, ins=None, coeff=1)
        has_dep, dist = gcd_dependence_test(acc1, acc2)
        self.assertFalse(has_dep)

    def test_vectorizability_pass_vector_add(self):
        src = """func main():
            n = 10
            a = new_array(n)
            b = new_array(n)
            c = new_array(n)
            i = 0
        loop:
            cond = i < n
            ifz cond goto done
            t1 = a[i]
            t2 = b[i]
            t3 = t1 + t2
            c[i] = t3
            i = i + 1
            goto loop
        done:
            return
        """
        fn = parse(src)[0]
        cfg = build_cfg(fn)
        loops = find_natural_loops(cfg)
        self.assertEqual(len(loops), 1)
        is_vec, reason, ctx = check_vectorizability(loops[0], cfg)
        self.assertTrue(is_vec)
        self.assertEqual(ctx["iv"], "i")
        self.assertEqual(ctx["limit"], "n")

    def test_vectorizability_fail_stride_not_one(self):
        src = """func main():
            a = new_array(10)
            i = 0
        loop:
            cond = i < 10
            ifz cond goto done
            a[i] = 1
            i = i + 2
            goto loop
        done:
            return
        """
        fn = parse(src)[0]
        cfg = build_cfg(fn)
        loops = find_natural_loops(cfg)
        is_vec, reason, ctx = check_vectorizability(loops[0], cfg)
        self.assertFalse(is_vec)
        self.assertIn("stride 2 != 1", reason)

    def test_vectorizability_fail_function_call(self):
        src = """func foo(x):
            return x

        func main():
            a = new_array(10)
            i = 0
        loop:
            cond = i < 10
            ifz cond goto done
            t = call foo(i)
            a[i] = t
            i = i + 1
            goto loop
        done:
            return
        """
        fns = parse(src)
        cfg = build_cfg(fns[1])
        loops = find_natural_loops(cfg)
        is_vec, reason, ctx = check_vectorizability(loops[0], cfg)
        self.assertFalse(is_vec)
        self.assertIn("function call 'foo'", reason)

    def test_vectorizability_fail_early_exit_multi_block(self):
        src = """func main():
            a = new_array(10)
            i = 0
        loop:
            cond = i < 10
            ifz cond goto done
            t = a[i]
            early = t == 99
            ifnz early goto done
            i = i + 1
            goto loop
        done:
            return
        """
        fn = parse(src)[0]
        cfg = build_cfg(fn)
        loops = find_natural_loops(cfg)
        is_vec, reason, ctx = check_vectorizability(loops[0], cfg)
        self.assertFalse(is_vec)
        self.assertTrue("multiple basic blocks" in reason or "exit" in reason)

    def test_vectorizability_fail_no_arrays(self):
        src = """func main():
            sum = 0
            i = 0
        loop:
            cond = i < 10
            ifz cond goto done
            sum = sum + i
            i = i + 1
            goto loop
        done:
            return sum
        """
        fn = parse(src)[0]
        cfg = build_cfg(fn)
        loops = find_natural_loops(cfg)
        is_vec, reason, ctx = check_vectorizability(loops[0], cfg)
        self.assertFalse(is_vec)
        self.assertIn("no array accesses found", reason)

    def test_vectorizability_fail_constant_index(self):
        src = """func main():
            a = new_array(10)
            i = 0
        loop:
            cond = i < 10
            ifz cond goto done
            t = a[0]
            a[i] = t
            i = i + 1
            goto loop
        done:
            return
        """
        fn = parse(src)[0]
        cfg = build_cfg(fn)
        loops = find_natural_loops(cfg)
        is_vec, reason, ctx = check_vectorizability(loops[0], cfg)
        self.assertFalse(is_vec)
        self.assertIn("constant", reason)

    def test_vector_ir_transformation(self):
        src = """func main():
            n = 10
            a = new_array(n)
            b = new_array(n)
            c = new_array(n)
            i = 0
        loop:
            cond = i < n
            ifz cond goto done
            t1 = a[i]
            t2 = b[i]
            t3 = t1 + t2
            c[i] = t3
            i = i + 1
            goto loop
        done:
            return
        """
        fns = parse(src)
        vec_fns, stats = vectorize_program(fns)
        self.assertEqual(stats.loops_vectorized, 1)

        vec_fn = vec_fns[0]
        types = [type(ins) for ins in vec_fn.body]
        self.assertIn(ir.VLoad, types)
        self.assertIn(ir.VStore, types)
        self.assertIn(ir.VAdd, types)
        self.assertIn(ir.LoadIndexed, types)
        self.assertIn(ir.StoreIndexed, types)

    def test_vm_vector_opcodes_direct(self):
        # Direct bytecode execution of all vector and indexed opcodes
        code = [
            Instr(Op.CALL, 2, 0),
            Instr(Op.HALT),
            Instr(Op.ENTER, 6),
            # slot 0: array A, slot 1: array B, slot 2: array C
            Instr(Op.PUSH, 8),
            Instr(Op.NEW_ARRAY),
            Instr(Op.STORE, 0),
            Instr(Op.PUSH, 8),
            Instr(Op.NEW_ARRAY),
            Instr(Op.STORE, 1),
            Instr(Op.PUSH, 8),
            Instr(Op.NEW_ARRAY),
            Instr(Op.STORE, 2),
            # Store scalar 10 into A[0], store vector [1,2,3,4] into B[0..3]
            Instr(Op.PUSH, 0),
            Instr(Op.PUSH, 10),
            Instr(Op.STORE_INDEXED, 0),
            Instr(Op.PUSH, 0),
            Instr(Op.PUSH, 20),
            Instr(Op.VSTORE, 1, 4),
            # VLOAD A[0..3], VLOAD B[0..3], VADD, VSTORE C[0..3]
            Instr(Op.PUSH, 0),
            Instr(Op.VLOAD, 0, 4),
            Instr(Op.PUSH, 0),
            Instr(Op.VLOAD, 1, 4),
            Instr(Op.VADD),
            Instr(Op.STORE, 3),
            Instr(Op.PUSH, 0),
            Instr(Op.LOAD, 3),
            Instr(Op.VSTORE, 2, 4),
            # LOAD_INDEXED C[0] -> 10 + 20 = 30
            Instr(Op.PUSH, 0),
            Instr(Op.LOAD_INDEXED, 2),
            Instr(Op.PRINT),
            # VMUL [2,2,2,2] * [3,3,3,3] -> [6,6,6,6]
            Instr(Op.PUSH, 0),
            Instr(Op.PUSH, 2),
            Instr(Op.VSTORE, 0, 4),
            Instr(Op.PUSH, 0),
            Instr(Op.PUSH, 3),
            Instr(Op.VSTORE, 1, 4),
            Instr(Op.PUSH, 0),
            Instr(Op.VLOAD, 0, 4),
            Instr(Op.PUSH, 0),
            Instr(Op.VLOAD, 1, 4),
            Instr(Op.VMUL),
            Instr(Op.STORE, 4),
            Instr(Op.PUSH, 0),
            Instr(Op.LOAD, 4),
            Instr(Op.VSTORE, 2, 4),
            Instr(Op.PUSH, 0),
            Instr(Op.LOAD_INDEXED, 2),
            Instr(Op.PRINT),
            Instr(Op.PUSH, 0),
            Instr(Op.RET),
        ]
        out = []
        vm = VM(code, out=out.append)
        vm.run()
        self.assertEqual(out, [30, 6])

    def test_end_to_end_vector_add_exact_multiple(self):
        # N = 8: exactly 2 vector iterations (width 4), 0 remainder iterations
        src = """func main():
            a = new_array(8)
            b = new_array(8)
            c = new_array(8)
            i = 0
        init:
            c1 = i < 8
            ifz c1 goto add
            a[i] = 10
            b[i] = 20
            i = i + 1
            goto init
        add:
            j = 0
        add_loop:
            c2 = j < 8
            ifz c2 goto done
            t1 = a[j]
            t2 = b[j]
            t3 = t1 + t2
            c[j] = t3
            j = j + 1
            goto add_loop
        done:
            r0 = c[0]
            print r0
            r7 = c[7]
            print r7
            return
        """
        fns = parse(src)
        vec_fns, stats = vectorize_program(fns)
        self.assertGreater(stats.loops_vectorized, 0)
        prog = generate(vec_fns)
        out = []
        vm = VM(prog.code, out=out.append)
        vm.run()
        self.assertEqual(out, [30, 30])

    def test_end_to_end_vector_add_with_remainder(self):
        # N = 10: 2 vector iterations (elements 0..7) + 2 remainder iterations (8..9)
        src = """func main():
            a = new_array(10)
            b = new_array(10)
            c = new_array(10)
            i = 0
        init:
            c1 = i < 10
            ifz c1 goto add
            a[i] = 7
            b[i] = 3
            i = i + 1
            goto init
        add:
            j = 0
        add_loop:
            c2 = j < 10
            ifz c2 goto done
            t1 = a[j]
            t2 = b[j]
            t3 = t1 + t2
            c[j] = t3
            j = j + 1
            goto add_loop
        done:
            r0 = c[0]
            print r0
            r8 = c[8]
            print r8
            r9 = c[9]
            print r9
            return
        """
        fns = parse(src)
        vec_fns, stats = vectorize_program(fns)
        self.assertGreater(stats.loops_vectorized, 0)
        prog = generate(vec_fns)
        out = []
        vm = VM(prog.code, out=out.append)
        vm.run()
        self.assertEqual(out, [10, 10, 10])

    def test_end_to_end_trip_count_less_than_width(self):
        # N = 3 (N < 4): 0 vector iterations, 3 remainder iterations
        src = """func main():
            a = new_array(3)
            b = new_array(3)
            c = new_array(3)
            i = 0
        init:
            c1 = i < 3
            ifz c1 goto add
            a[i] = 100
            b[i] = 50
            i = i + 1
            goto init
        add:
            j = 0
        add_loop:
            c2 = j < 3
            ifz c2 goto done
            t1 = a[j]
            t2 = b[j]
            t3 = t1 - t2
            c[j] = t3
            j = j + 1
            goto add_loop
        done:
            r0 = c[0]
            print r0
            r2 = c[2]
            print r2
            return
        """
        fns = parse(src)
        vec_fns, stats = vectorize_program(fns)
        prog = generate(vec_fns)
        out = []
        vm = VM(prog.code, out=out.append)
        vm.run()
        self.assertEqual(out, [50, 50])

    def test_end_to_end_vector_ops_sub_mul_div(self):
        src = """func main():
            a = new_array(8)
            b = new_array(8)
            c = new_array(8)
            i = 0
        init:
            c1 = i < 8
            ifz c1 goto mul_loop
            a[i] = 40
            b[i] = 4
            i = i + 1
            goto init
        mul_loop:
            j = 0
        mul_step:
            c2 = j < 8
            ifz c2 goto done
            t1 = a[j]
            t2 = b[j]
            t3 = t1 / t2
            c[j] = t3
            j = j + 1
            goto mul_step
        done:
            ans = c[3]
            print ans
            return
        """
        fns = parse(src)
        vec_fns, stats = vectorize_program(fns)
        prog = generate(vec_fns)
        out = []
        vm = VM(prog.code, out=out.append)
        vm.run()
        self.assertEqual(out, [10])

    def test_vectorization_stats_report(self):
        stats = VectorizationStats(
            loops_analyzed=3,
            loops_vectorized=2,
            details=[
                LoopVectorizationResult(
                    header="loop1",
                    vectorized=True,
                    reason="vectorized with SIMD width 4 (remainder loop generated)",
                    speedup=2.5,
                ),
                LoopVectorizationResult(
                    header="loop2",
                    vectorized=False,
                    reason="no array accesses found in loop",
                    speedup=1.0,
                ),
            ],
        )
        report = stats.format_report()
        self.assertIn("VECTORIZATION REPORT", report)
        self.assertIn("Loops analyzed:              3", report)
        self.assertIn("Loops vectorized:            2", report)
        self.assertIn("Loop 'loop1': VECTORIZED", report)
        self.assertIn("Speedup: 2.50x", report)
        self.assertIn("Loop 'loop2': NOT VECTORIZED", report)
        self.assertIn("no array accesses found", report)


# =====================================================================
# Phase 8: AI Pass Selector Tests
# =====================================================================

import os
import ai_selector
from ai_selector import (
    AIPassSelector,
    AISelectorResult,
    PassDecision,
    PassScoringModel,
    PassSequenceGenerator,
    ProgramFeatureExtractor,
    ProgramFeatures,
    ai_select_passes,
    load_history,
    record_history,
    run_ai_pipeline,
)


class AIPassSelectorTests(unittest.TestCase):
    def test_feature_extraction_structural(self):
        src = """func main():
            i = 0
            n = 10
        outer:
            c1 = i < n
            ifz c1 goto done
            j = 0
        inner:
            c2 = j < n
            ifz c2 goto outer_step
            j = j + 1
            goto inner
        outer_step:
            i = i + 1
            goto outer
        done:
            return
        """
        fns = parse(src)
        feat = ProgramFeatureExtractor().extract(fns)
        self.assertEqual(feat.function_count, 1)
        self.assertEqual(feat.loop_count, 2)
        self.assertEqual(feat.max_loop_nesting_depth, 2)
        self.assertTrue(feat.has_inner_loops)
        self.assertGreater(feat.branch_density, 0.0)
        self.assertGreater(feat.avg_block_size, 0.0)

    def test_feature_extraction_expressions(self):
        src = """func main():
            a = 10
            b = 20
            c = a + b
            d = c * 2
            e = d / 4
            x = a
            return e
        """
        fns = parse(src)
        feat = ProgramFeatureExtractor().extract(fns)
        self.assertGreater(feat.const_literal_fraction, 0.0)
        self.assertGreater(feat.copy_instruction_fraction, 0.0)
        self.assertGreater(feat.arithmetic_intensity, 0.0)
        self.assertGreater(feat.multiply_divide_count, 0)

    def test_feature_extraction_variables(self):
        src = """func main():
            a = 1
            b = 2
            c = a + b
            dead1 = 99
            dead2 = 100
            print c
            return
        """
        fns = parse(src)
        feat = ProgramFeatureExtractor().extract(fns)
        self.assertGreater(feat.max_live_variables, 0)
        self.assertGreater(feat.variable_reuse_rate, 0.0)
        self.assertGreater(feat.dead_variable_fraction, 0.0)

    def test_feature_extraction_loop_categories(self):
        src = """func main():
            a = new_array(10)
            i = 0
        loop:
            c = i < 10
            ifz c goto done
            a[i] = 42
            i = i + 1
            goto loop
        done:
            return
        """
        fns = parse(src)
        feat = ProgramFeatureExtractor().extract(fns)
        self.assertEqual(feat.loops_with_array_access, 1)
        self.assertEqual(feat.loops_with_induction_vars, 1)
        self.assertEqual(feat.loops_with_known_count, 1)
        self.assertGreater(feat.array_access_count, 0)

    def test_scoring_rules_and_ranges(self):
        feat = ProgramFeatures(
            total_instructions=20,
            loop_count=2,
            loops_with_array_access=2,
            const_literal_fraction=0.4,
            dead_variable_fraction=0.3,
            copy_instruction_fraction=0.2,
            max_live_variables=10,
            arithmetic_intensity=2.5,
        )
        scorer = PassScoringModel()
        decisions = scorer.score(feat, history_file="non_existent.json")

        for name, dec in decisions.items():
            self.assertGreaterEqual(dec.score, 0.0)
            self.assertLessEqual(dec.score, 1.0)
            self.assertIsInstance(dec.reason, str)
            self.assertGreater(len(dec.reason), 0)

        # Vectorizer and ConstFold should have high expected scores
        self.assertGreater(decisions["Vectorizer"].score, 0.5)
        self.assertGreater(decisions["ConstantFolding"].score, 0.3)
        self.assertGreater(decisions["RegisterAllocation"].score, 0.8)

    def test_pass_ordering_constraints(self):
        gen = PassSequenceGenerator()
        decisions = {
            "InstructionScheduler": PassDecision("InstructionScheduler", 0.9, 0.9, "sched", False),
            "RegisterAllocation": PassDecision("RegisterAllocation", 0.8, 0.8, "reg", False),
            "Vectorizer": PassDecision("Vectorizer", 0.85, 0.85, "vec", False),
            "ConstantFolding": PassDecision("ConstantFolding", 0.7, 0.7, "const", False),
            "LICM": PassDecision("LICM", 0.6, 0.6, "licm", False),
        }
        ordered, skipped = gen.generate(decisions, threshold=0.2)

        # Ordering constraints:
        # LICM -> ConstantFolding -> Vectorizer -> RegisterAllocation -> InstructionScheduler
        self.assertLess(ordered.index("LICM"), ordered.index("ConstantFolding"))
        self.assertLess(ordered.index("ConstantFolding"), ordered.index("Vectorizer"))
        self.assertLess(ordered.index("Vectorizer"), ordered.index("RegisterAllocation"))
        self.assertLess(ordered.index("RegisterAllocation"), ordered.index("InstructionScheduler"))

    def test_prerequisite_forcing(self):
        gen = PassSequenceGenerator()
        decisions = {
            # InstructionScheduler scores high, but RegisterAllocation scores below threshold
            "InstructionScheduler": PassDecision("InstructionScheduler", 0.8, 0.8, "sched", False),
            "RegisterAllocation": PassDecision("RegisterAllocation", 0.1, 0.1, "reg", False),
        }
        ordered, skipped = gen.generate(decisions, threshold=0.2)
        # RegisterAllocation must be forced as prerequisite before InstructionScheduler
        self.assertIn("RegisterAllocation", ordered)
        self.assertIn("InstructionScheduler", ordered)
        self.assertLess(ordered.index("RegisterAllocation"), ordered.index("InstructionScheduler"))
        self.assertTrue(decisions["RegisterAllocation"].is_prerequisite)

    def test_feedback_loop_learning(self):
        tmp_file = "test_feedback_history.json"
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

        feat = ProgramFeatures(
            total_instructions=15,
            const_literal_fraction=0.3,
        )
        scorer = PassScoringModel()
        d_init = scorer.score(feat, history_file=tmp_file)
        init_score = d_init["ConstantFolding"].score

        # Record positive outcome runs
        record_history(
            features=feat.to_dict(),
            passes_run=["ConstantFolding"],
            instructions_before=15,
            instructions_after=8,
            speedup=1.8,
            history_file=tmp_file,
        )

        d_after = scorer.score(feat, history_file=tmp_file)
        learned_score = d_after["ConstantFolding"].score

        self.assertGreater(learned_score, init_score)

        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    def test_explainability_report_content(self):
        src = """func main():
            a = 10
            b = 20
            c = a + b
            print c
            return
        """
        fns = parse(src)
        res = ai_select_passes(fns)
        rep = res.format_report()
        self.assertIn("AI Pass Selector Report", rep)
        self.assertIn("Program Profile:", rep)
        self.assertIn("Selected Passes (in order):", rep)
        self.assertIn("Skipped Passes:", rep)

    def test_ai_pipeline_execution_sum(self):
        src = """func main():
            n = 10
            sum = 0
            i = 1
        loop:
            c = i > n
            ifnz c goto done
            sum = sum + i
            i = i + 1
            goto loop
        done:
            print sum
            return
        """
        fns = parse(src)
        res = ai_select_passes(fns)
        opt_fns, reg_map, reg_stats, sched_stats, vec_stats, stats = run_ai_pipeline(
            fns, res.selected_passes
        )
        prog = generate(opt_fns, reg_map=reg_map)
        out = []
        vm = VM(prog.code, out=out.append)
        vm.run()
        self.assertEqual(out, [55])

    def test_ai_pipeline_execution_vector_add(self):
        src = """func main():
            a = new_array(8)
            b = new_array(8)
            c = new_array(8)
            i = 0
        init:
            c1 = i < 8
            ifz c1 goto add
            a[i] = 10
            b[i] = 20
            i = i + 1
            goto init
        add:
            j = 0
        add_loop:
            c2 = j < 8
            ifz c2 goto done
            t1 = a[j]
            t2 = b[j]
            t3 = t1 + t2
            c[j] = t3
            j = j + 1
            goto add_loop
        done:
            r0 = c[0]
            print r0
            r7 = c[7]
            print r7
            return
        """
        fns = parse(src)
        res = ai_select_passes(fns)
        self.assertIn("Vectorizer", res.selected_passes)
        opt_fns, reg_map, reg_stats, sched_stats, vec_stats, stats = run_ai_pipeline(
            fns, res.selected_passes
        )
        prog = generate(opt_fns, reg_map=reg_map)
        out = []
        vm = VM(prog.code, out=out.append)
        vm.run()
        self.assertEqual(out, [30, 30])


class BenchmarkTests(unittest.TestCase):
    """Test suite for Phase 9: Benchmark Comparison Engine."""

    def test_profiling_vm_counters(self):
        """Test that ProfilingVM tracks peak stack depth, memory accesses, and calls."""
        from isa import Instr, Op
        from benchmark import ProfilingVM

        code = [
            Instr(Op.CALL, 2, 0),
            Instr(Op.HALT),
            Instr(Op.ENTER, 2),
            Instr(Op.PUSH, 10),
            Instr(Op.PUSH, 20),
            Instr(Op.STORE, 0),
            Instr(Op.LOAD, 0),
            Instr(Op.PRINT),
            Instr(Op.PUSH, 0),
            Instr(Op.RET),
        ]
        pvm = ProfilingVM(code)
        output = pvm.run()
        self.assertEqual(output, [20])
        self.assertGreaterEqual(pvm.steps, 8)
        self.assertGreaterEqual(pvm.peak_stack_depth, 2)
        self.assertEqual(pvm.memory_accesses, 2)
        self.assertEqual(pvm.call_count, 1)

    def test_benchmark_metrics_collection(self):
        """Test dual-compilation metrics collection on sum.ir."""
        from benchmark import BenchmarkRunner, BenchmarkMetrics

        runner = BenchmarkRunner("examples/sum.ir", timer_runs=3, verbose=False)
        metrics = runner.run()

        self.assertIsInstance(metrics, BenchmarkMetrics)
        self.assertTrue(metrics.correctness)
        self.assertEqual(metrics.baseline_output, [55])
        self.assertEqual(metrics.optimized_output, [55])

        # Static metrics
        self.assertGreater(metrics.instruction_count_before, 0)
        self.assertGreater(metrics.instruction_count_after, 0)
        self.assertGreater(metrics.basic_block_count_before, 0)
        self.assertGreater(metrics.basic_block_count_after, 0)

        # Dynamic metrics
        self.assertGreater(metrics.vm_instructions_executed_before, 0)
        self.assertGreater(metrics.vm_instructions_executed_after, 0)
        self.assertGreater(metrics.execution_time_ms_before, 0.0)
        self.assertGreater(metrics.execution_time_ms_after, 0.0)
        self.assertGreater(metrics.operand_stack_peak_depth_before, 0)
        self.assertGreater(metrics.operand_stack_peak_depth_after, 0)
        self.assertGreater(metrics.estimated_cache_misses_before, 0)

        # Derived metrics
        self.assertGreaterEqual(metrics.static_speedup, 0.0)
        self.assertGreater(metrics.dynamic_speedup, 0.0)
        self.assertGreater(metrics.wall_speedup, 0.0)

    def test_correctness_verification_success(self):
        """Test output equivalence verification passes on constant_expr.ir."""
        from benchmark import BenchmarkRunner

        runner = BenchmarkRunner("examples/constant_expr.ir", timer_runs=2, verbose=False)
        metrics = runner.run()
        self.assertTrue(metrics.correctness)
        self.assertEqual(metrics.baseline_output, metrics.optimized_output)

    def test_correctness_verification_failure(self):
        """Test that an output mismatch triggers CorrectnessError and aborts."""
        import tempfile
        from benchmark import BenchmarkRunner, CorrectnessError

        # Create a test runner and monkeypatch VM execution to produce differing outputs
        runner = BenchmarkRunner("examples/constant_expr.ir", timer_runs=1, verbose=False)

        orig_measure = runner._measure_wall_time
        try:
            from benchmark import ProfilingVM
            orig_run = ProfilingVM.run

            def buggy_run(self):
                # If this is the optimized VM, inject incorrect output
                res = orig_run(self)
                if len(self.recorded_output) > 0 and self.steps < 30:
                    self.recorded_output[0] = 999999
                return res

            ProfilingVM.run = buggy_run
            with self.assertRaises(CorrectnessError) as ctx:
                runner.run()
            self.assertIn("CORRECTNESS FAILURE", str(ctx.exception))
        finally:
            ProfilingVM.run = orig_run

    def test_report_file_generation(self):
        """Test JSON and Markdown report export to reports directory."""
        import tempfile
        import json
        from benchmark import BenchmarkRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(
                "examples/copy_chain.ir",
                reports_dir=tmpdir,
                timer_runs=2,
                verbose=False,
            )
            metrics = runner.run()

            files = os.listdir(tmpdir)
            json_files = [f for f in files if f.endswith(".json")]
            md_files = [f for f in files if f.endswith(".md")]

            self.assertEqual(len(json_files), 1)
            self.assertEqual(len(md_files), 1)

            # Validate JSON content
            with open(os.path.join(tmpdir, json_files[0]), "r") as f:
                data = json.load(f)
            self.assertEqual(data["program"], "examples/copy_chain.ir")
            self.assertTrue(data["correctness"])
            self.assertIn("static_metrics", data)
            self.assertIn("dynamic_metrics", data)
            self.assertIn("derived_metrics", data)

            # Validate Markdown content
            with open(os.path.join(tmpdir, md_files[0]), "r") as f:
                md_text = f.read()
            self.assertIn("# CVM Optimization Benchmark Report", md_text)
            self.assertIn("## Static Analysis", md_text)
            self.assertIn("## Dynamic Execution", md_text)
            self.assertIn("## Pass-by-Pass Contribution", md_text)

    def test_derived_metrics_math(self):
        """Test derived speedup and reduction math calculation."""
        from benchmark import BenchmarkMetrics

        m = BenchmarkMetrics(
            program="test.ir",
            timestamp="2026-10-21",
            correctness=True,
            baseline_output=[1],
            optimized_output=[1],
            selected_passes=["ConstFold"],
            instruction_count_before=100,
            instruction_count_after=50,
            basic_block_count_before=10,
            basic_block_count_after=8,
            loop_count_before=1,
            loop_count_after=1,
            dead_code_removed=20,
            constants_folded=30,
            copies_eliminated=0,
            cse_replacements=0,
            loops_vectorized=0,
            instructions_hoisted_by_LICM=0,
            estimated_register_spills=0,
            vm_instructions_executed_before=1000,
            vm_instructions_executed_after=500,
            execution_time_ms_before=2.0,
            execution_time_ms_after=1.0,
            operand_stack_peak_depth_before=10,
            operand_stack_peak_depth_after=5,
            estimated_cache_misses_before=100,
            estimated_cache_misses_after=50,
            function_call_count_before=0,
            function_call_count_after=0,
            static_speedup=2.0,
            dynamic_speedup=2.0,
            wall_speedup=2.0,
            code_size_reduction_percent=50.0,
        )
        d = m.to_dict()
        self.assertEqual(d["derived_metrics"]["static_speedup"], 2.0)
        self.assertEqual(d["derived_metrics"]["dynamic_speedup"], 2.0)
        self.assertEqual(d["derived_metrics"]["wall_speedup"], 2.0)
        self.assertEqual(d["derived_metrics"]["code_size_reduction_percent"], 50.0)

    def test_all_examples_regression_suite(self):
        """Test that run_regression across examples/ returns 0 with no regressions."""
        from benchmark import run_regression

        ret = run_regression("examples", verbose=False)
        self.assertEqual(ret, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


