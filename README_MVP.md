# Robin Logistics MVP - Client Demo Ready

## What This MVP Does
• **Inventory-aware greedy solver** with multi-warehouse pickups and partial order fulfillment
• **Geographic proximity optimization** using Haversine distance for efficient route planning  
• **Capacity-constrained routing** respecting vehicle weight/volume limits with safety margins
• **Multi-page dashboard** for comprehensive solution visualization and analysis
• **Robust validation** with graceful fallbacks for invalid routes and edge cases

## Quick Setup

### 1. Install Dependencies
```bash
# Core dependency
pip install robin-logistics-env

# For dashboard (optional)
pip install streamlit

# Upgrade to latest version if needed
pip install --upgrade robin-logistics-env
```

### 2. Verify Installation
```bash
python tests/test_solution_schema.py
```
This runs basic schema validation tests and confirms everything works.

## Running the Project

### Method 1: Headless Execution (Recommended for Testing)
```bash
python run_headless_local.py
```
**What it does:**
- Runs solver without GUI
- Shows validation results
- Displays fulfillment summary and costs  
- Exits with code 0 if valid, non-zero if invalid

### Method 2: Existing Environment Dashboard
```bash
python run_dashboard.py
```
**What it does:**
- Uses the original dashboard from robin-logistics-env
- Calls `my_solver(env)` function
- Standard environment interface with basic visualization

### Method 3: MVP Client Dashboard (New)
```bash
streamlit run mvp_dashboard.py
```
**What it does:**
- Polished multi-page client demo
- Overview, fulfillment analysis, route details, orders, inventory, map
- Geographic visualization with lat/lon plotting
- Comprehensive KPIs and metrics

### Method 4: Manual Testing
```bash
python -c "
from robin_logistics import LogisticsEnvironment
import solver
env = LogisticsEnvironment()
solution = solver.solver(env)
is_valid, msg = env.validate_solution_complete(solution)
print(f'Valid: {is_valid}, Routes: {len(solution[\"routes\"])}, Message: {msg}')
"
```

## Testing & Validation

### Run All Tests
```bash
python tests/test_solution_schema.py
```
Tests include:
- Solver import and function availability
- Environment initialization
- Solution structure validation (routes, steps, operations)
- Home warehouse start/end constraint
- Full solution validation via environment
- Consistency between `solver()` and `my_solver()`

### Enable Verbose Logging
```bash
export MVP_LOG=1
python run_headless_local.py
```
Or in the dashboard, check "Enable Verbose Logging"

### Manual Validation Steps
1. **Schema Check**: Ensure solution has `{"routes": [{"vehicle_id": "...", "steps": [...]}]}`
2. **Route Validation**: Each route starts and ends at vehicle's home warehouse node
3. **Operations Check**: Pickups have warehouse_id, deliveries have order_id, quantities > 0
4. **Capacity Limits**: No vehicle exceeds weight/volume constraints
5. **Inventory Check**: No warehouse picked more items than available

## Key Constraints & Behavior

### Route Requirements
- **Start/End**: Every route begins and ends at vehicle's home warehouse node
- **Connectivity**: Sequential nodes must be connected in road network
- **Operations**: Pickups only at warehouse nodes, deliveries only at order destination nodes
- **One Route Per Vehicle**: Each vehicle gets exactly one route (may be empty)

### Solver Strategy
- **Fulfillment Priority**: Maximizes items delivered before optimizing cost
- **Multi-warehouse**: Can pick from multiple warehouses for single order
- **Capacity Aware**: Uses 95% of capacity with safety margins
- **Geographic**: Prioritizes nearby orders and warehouses using Haversine distance
- **Graceful Degradation**: Falls back to minimal routes if validation fails

### Performance Limits
- Max 50 steps per vehicle route
- Max 20 orders per vehicle  
- Max 5 warehouses per order
- Capacity safety margin of 95%

## Submission Preparation

When ready to submit, rename the solver file:
```bash
cp solver.py {TEAM_NAME}_solver_1.py
```

The evaluation system will call `solver(env)` function from your renamed file.

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
pip install --upgrade robin-logistics-env
# Ensure you're in the correct directory
```

**Dashboard Won't Start:**
```bash
# Try different port
streamlit run mvp_dashboard.py --server.port 8502
```

**Empty Routes:**
- Check if orders exist: `env.get_all_order_ids()`
- Check vehicle availability: `env.get_available_vehicles()`  
- Enable logging: `export MVP_LOG=1`

**Validation Failures:**
- Routes must start/end at home warehouse
- Check inventory availability at warehouses
- Verify node connectivity in road network
- Operations must match node types (warehouse vs order destination)

### Debug Mode
```bash
python -c "
import os
os.environ['MVP_LOG'] = '1'
from robin_logistics import LogisticsEnvironment
import solver
env = LogisticsEnvironment()
print('Orders:', len(env.get_all_order_ids()))
print('Vehicles:', len(env.get_available_vehicles()))
print('Warehouses:', len(env.warehouses))
solution = solver.solver(env)
print('Solution routes:', len(solution.get('routes', [])))
"
```

### Performance Monitoring
- Check fulfillment rate (should be >80% for good performance)
- Monitor cost efficiency (lower is better)
- Verify reasonable route complexity (steps per vehicle)
- Watch for timeout issues (solver should complete <30 seconds)

## FAQ

**Q: Can I modify the existing files?**
A: No, only modify the 5 generated MVP files. Keep `run_dashboard.py` and other existing files unchanged.

**Q: Which dashboard should I use for demos?**
A: Use `mvp_dashboard.py` for client presentations (polished, multi-page). Use `run_dashboard.py` for development/testing.

**Q: How do I know if my solution is good?**
A: Check fulfillment rate >80%, reasonable cost, and valid validation. The scoring formula penalizes unfulfilled orders heavily.

**Q: What if solver takes too long?**
A: Adjust limits in solver.py: reduce `ORDER_PER_VEHICLE_LIMIT`, `MAX_STEPS_PER_VEHICLE`, or `MAX_WAREHOUSES_PER_ORDER`.

**Q: Can I use external libraries?**
A: Only for the dashboard (`streamlit`). The solver must use only standard library to ensure submission compatibility.
