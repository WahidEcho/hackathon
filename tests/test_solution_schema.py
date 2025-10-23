#!/usr/bin/env python3
"""
Basic schema validation tests for Robin Logistics MVP solver.
Run with: python -m tests.test_solution_schema
"""

import sys
import os
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_solver_import():
    """Test that solver can be imported successfully."""
    print("Testing solver import...")
    try:
        import solver
        assert hasattr(solver, 'solver'), "solver.solver function not found"
        assert hasattr(solver, 'my_solver'), "solver.my_solver function not found"
        print("✅ Solver import test passed")
        return True
    except Exception as e:
        print(f"❌ Solver import test failed: {e}")
        return False


def test_environment_initialization():
    """Test that environment can be initialized."""
    print("Testing environment initialization...")
    try:
        from robin_logistics import LogisticsEnvironment
        env = LogisticsEnvironment()
        assert env is not None, "Environment is None"
        print("✅ Environment initialization test passed")
        return env
    except Exception as e:
        print(f"❌ Environment initialization test failed: {e}")
        return None


def test_basic_solution_structure(solution):
    """Test basic solution structure requirements."""
    print("Testing basic solution structure...")
    try:
        # Solution should be a dict
        assert isinstance(solution, dict), f"Solution should be dict, got {type(solution)}"
        
        # Should have 'routes' key
        assert 'routes' in solution, "Solution missing 'routes' key"
        
        # Routes should be a list
        routes = solution['routes']
        assert isinstance(routes, list), f"Routes should be list, got {type(routes)}"
        
        print(f"✅ Solution has {len(routes)} routes")
        
        # Test each route structure
        for i, route in enumerate(routes):
            assert isinstance(route, dict), f"Route {i} should be dict, got {type(route)}"
            assert 'vehicle_id' in route, f"Route {i} missing 'vehicle_id'"
            assert 'steps' in route, f"Route {i} missing 'steps'"
            
            steps = route['steps']
            assert isinstance(steps, list), f"Route {i} steps should be list, got {type(steps)}"
            
            # Test each step structure
            for j, step in enumerate(steps):
                assert isinstance(step, dict), f"Route {i} step {j} should be dict"
                assert 'node_id' in step, f"Route {i} step {j} missing 'node_id'"
                assert 'pickups' in step, f"Route {i} step {j} missing 'pickups'"
                assert 'deliveries' in step, f"Route {i} step {j} missing 'deliveries'"
                assert 'unloads' in step, f"Route {i} step {j} missing 'unloads'"
        
        print("✅ Basic solution structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic solution structure test failed: {e}")
        return False


def test_solution_validation(solution, env):
    """Test solution validation through environment."""
    print("Testing solution validation...")
    try:
        is_valid, message = env.validate_solution_complete(solution)
        
        if is_valid:
            print("✅ Solution validation test passed - solution is VALID")
        else:
            print(f"⚠️  Solution validation test - solution is INVALID: {message}")
            print("Note: This may be expected for some test scenarios")
        
        return is_valid
        
    except Exception as e:
        print(f"❌ Solution validation test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("Robin Logistics MVP - Solution Schema Tests")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Solver import
    total_tests += 1
    if test_solver_import():
        tests_passed += 1
    
    # Test 2: Environment initialization
    total_tests += 1
    env = test_environment_initialization()
    if env:
        tests_passed += 1
    else:
        print("❌ Cannot continue tests without environment")
        return tests_passed, total_tests
    
    # Test 3: Generate solution
    print("Generating solution for testing...")
    try:
        import solver
        solution = solver.solver(env)
        print(f"✅ Solution generated with {len(solution.get('routes', []))} routes")
    except Exception as e:
        print(f"❌ Solution generation failed: {e}")
        return tests_passed, total_tests
    
    # Test 4: Basic solution structure
    total_tests += 1
    if test_basic_solution_structure(solution):
        tests_passed += 1
    
    # Test 5: Solution validation
    total_tests += 1
    if test_solution_validation(solution, env):
        tests_passed += 1
    
    return tests_passed, total_tests


def main():
    """Main test execution."""
    try:
        passed, total = run_all_tests()
        
        print("\n" + "=" * 50)
        print(f"TEST RESULTS: {passed}/{total} tests passed")
        print("=" * 50)
        
        if passed == total:
            print("🎉 All tests passed!")
            return 0
        else:
            print(f"⚠️  {total - passed} test(s) failed")
            return 1
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
