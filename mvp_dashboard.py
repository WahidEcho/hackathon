#!/usr/bin/env python3
"""
MVP Dashboard for Robin Logistics Environment - Client Demo
Multi-page Streamlit application for visualizing solver performance.
"""

import streamlit as st
import json
import traceback
from typing import Dict, List, Any, Optional


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="Robin Logistics MVP Dashboard",
        page_icon="🚚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚚 Robin Logistics MVP Dashboard")
    st.markdown("*Multi-Warehouse Vehicle Routing Problem (MWVRP) Solver Demo*")
    
    # Initialize session state
    if 'solution' not in st.session_state:
        st.session_state.solution = None
    if 'env' not in st.session_state:
        st.session_state.env = None
    if 'validation_result' not in st.session_state:
        st.session_state.validation_result = None
    if 'solver_stats' not in st.session_state:
        st.session_state.solver_stats = None
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Fulfillment & Cost", "Vehicles & Routes", "Orders", "Warehouses & Inventory", "Map View"],
        index=0
    )
    
    # Page routing
    if page == "Overview":
        show_overview_page()
    elif page == "Fulfillment & Cost":
        show_fulfillment_cost_page()
    elif page == "Vehicles & Routes":
        show_vehicles_routes_page()
    elif page == "Orders":
        show_orders_page()
    elif page == "Warehouses & Inventory":
        show_warehouses_inventory_page()
    elif page == "Map View":
        show_map_page()


def show_overview_page():
    """Overview page with scenario summary and solver controls."""
    st.header("📊 Scenario Overview")
    
    # Environment initialization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Environment Setup")
        
        if st.button("🔄 Initialize New Scenario", type="primary"):
            try:
                with st.spinner("Initializing environment..."):
                    from robin_logistics import LogisticsEnvironment
                    st.session_state.env = LogisticsEnvironment()
                    st.session_state.solution = None
                    st.session_state.validation_result = None
                    st.session_state.solver_stats = None
                st.success("✅ Environment initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize environment: {str(e)}")
                st.code(traceback.format_exc())
    
    with col2:
        st.subheader("Solver Options")
        verbose_logging = st.checkbox("Enable Verbose Logging", value=False)
        if verbose_logging:
            import os
            os.environ['MVP_LOG'] = '1'
        else:
            import os
            os.environ.pop('MVP_LOG', None)
    
    # Environment info
    if st.session_state.env:
        st.subheader("📋 Scenario Summary")
        
        try:
            env = st.session_state.env
            
            # Collect basic stats
            orders = env.get_all_order_ids()
            vehicles = env.get_available_vehicles()
            warehouses = env.warehouses
            skus = env.skus
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Orders", len(orders))
            with col2:
                st.metric("Vehicles", len(vehicles))
            with col3:
                st.metric("Warehouses", len(warehouses))
            with col4:
                st.metric("SKU Types", len(skus))
            
            # Detailed breakdowns
            st.subheader("🏢 Warehouse Details")
            warehouse_data = []
            for wh_id, wh in warehouses.items():
                try:
                    inventory = env.get_warehouse_inventory(wh_id)
                    total_items = sum(inventory.values()) if inventory else 0
                    unique_skus = len(inventory) if inventory else 0
                    
                    warehouse_data.append({
                        "Warehouse ID": wh_id,
                        "Location Node": wh.location.id if wh.location else "N/A",
                        "Total Items": total_items,
                        "Unique SKUs": unique_skus,
                        "Vehicle Count": len(wh.vehicles) if hasattr(wh, 'vehicles') else 0
                    })
                except Exception as e:
                    warehouse_data.append({
                        "Warehouse ID": wh_id,
                        "Location Node": "Error",
                        "Total Items": 0,
                        "Unique SKUs": 0,
                        "Vehicle Count": 0
                    })
            
            st.dataframe(warehouse_data, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading scenario details: {str(e)}")
    
    # Solver execution
    if st.session_state.env:
        st.subheader("🎯 Run MVP Solver")
        
        if st.button("▶️ Execute Solver", type="primary", use_container_width=True):
            try:
                with st.spinner("Running MVP solver..."):
                    import solver
                    solution = solver.my_solver(st.session_state.env)
                    st.session_state.solution = solution
                    
                    # Validate solution
                    is_valid, message = st.session_state.env.validate_solution_complete(solution)
                    st.session_state.validation_result = (is_valid, message)
                    
                    # Collect stats
                    try:
                        fulfillment = st.session_state.env.get_solution_fulfillment_summary(solution)
                        cost = st.session_state.env.calculate_solution_cost(solution)
                        
                        st.session_state.solver_stats = {
                            'fulfillment': fulfillment,
                            'cost': cost
                        }
                    except Exception as stats_error:
                        st.warning(f"Could not collect detailed stats: {stats_error}")
                        st.session_state.solver_stats = None
                
                # Show immediate results
                is_valid, message = st.session_state.validation_result
                if is_valid:
                    st.success(f"✅ Solution generated successfully!")
                    st.info(f"Generated {len(solution.get('routes', []))} routes")
                else:
                    st.warning(f"⚠️ Solution generated but validation failed: {message}")
                
            except Exception as e:
                st.error(f"❌ Solver execution failed: {str(e)}")
                st.code(traceback.format_exc())
    else:
        st.info("👆 Please initialize the environment first to run the solver.")


def show_fulfillment_cost_page():
    """Fulfillment and cost analysis page."""
    st.header("📈 Fulfillment & Cost Analysis")
    
    if not st.session_state.solution:
        st.info("🔍 Please run the solver from the Overview page first.")
        return
    
    # Validation status
    if st.session_state.validation_result:
        is_valid, message = st.session_state.validation_result
        if is_valid:
            st.success("✅ Solution is valid")
        else:
            st.error(f"❌ Solution validation failed: {message}")
    
    # Key metrics
    if st.session_state.solver_stats:
        stats = st.session_state.solver_stats
        
        st.subheader("📊 Key Performance Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Cost",
                f"${stats['cost']:.2f}" if 'cost' in stats else "N/A"
            )
        
        with col2:
            fulfillment_pct = 0
            if 'fulfillment' in stats and stats['fulfillment']:
                fulfillment = stats['fulfillment']
                total_requested = fulfillment.get('total_requested', 0)
                total_delivered = fulfillment.get('total_delivered', 0)
                if total_requested > 0:
                    fulfillment_pct = (total_delivered / total_requested) * 100
            
            st.metric(
                "Fulfillment Rate",
                f"{fulfillment_pct:.1f}%"
            )
        
        with col3:
            active_routes = len([r for r in st.session_state.solution.get('routes', []) if len(r.get('steps', [])) > 1])
            st.metric("Active Routes", active_routes)


def show_vehicles_routes_page():
    """Vehicles and routes detailed view."""
    st.header("🚛 Vehicles & Routes")
    
    if not st.session_state.solution:
        st.info("🔍 Please run the solver from the Overview page first.")
        return
    
    solution = st.session_state.solution
    routes = solution.get('routes', [])
    
    if not routes:
        st.warning("⚠️ No routes found in solution.")
        return
    
    # Route summary
    st.subheader("📋 Route Summary")
    
    route_data = []
    for route in routes:
        vehicle_id = route.get('vehicle_id', 'Unknown')
        steps = route.get('steps', [])
        
        # Count operations
        total_pickups = sum(len(step.get('pickups', [])) for step in steps)
        total_deliveries = sum(len(step.get('deliveries', [])) for step in steps)
        
        route_data.append({
            "Vehicle ID": vehicle_id,
            "Steps": len(steps),
            "Pickups": total_pickups,
            "Deliveries": total_deliveries
        })
    
    st.dataframe(route_data, use_container_width=True)


def show_orders_page():
    """Orders and fulfillment status page."""
    st.header("📦 Orders Analysis")
    
    if not st.session_state.env:
        st.info("🔍 Please initialize the environment from the Overview page first.")
        return
    
    env = st.session_state.env
    order_ids = env.get_all_order_ids()
    
    st.subheader("📋 Order Summary")
    st.metric("Total Orders", len(order_ids))


def show_warehouses_inventory_page():
    """Warehouses and inventory analysis page."""
    st.header("🏢 Warehouses & Inventory")
    
    if not st.session_state.env:
        st.info("🔍 Please initialize the environment from the Overview page first.")
        return


def show_map_page():
    """Geographic visualization page."""
    st.header("🗺️ Geographic View")
    
    if not st.session_state.env:
        st.info("🔍 Please initialize the environment from the Overview page first.")
        return
    
    st.info("ℹ️ This is an abstracted geographic view using straight-line connections.")


if __name__ == "__main__":
    main()
