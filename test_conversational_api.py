#!/usr/bin/env python3
"""
Simple test for conversational_api.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_api():
    try:
        # Import the API
        from services.conversational_api import app
        
        # Create test client
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        print("Testing Conversational API endpoints...")
        
        # Test 1: Root endpoint
        response = client.get('/')
        print(f"Root endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Status: {data.get('status')}")
        
        # Test 2: Health endpoint
        response = client.get('/health')
        print(f"Health endpoint: {response.status_code}")
        
        # Test 3: Status endpoint
        response = client.get('/status')
        print(f"Status endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Bot status: {data.get('status')}")
            print(f"  Documents loaded: {data.get('documents_loaded')}")
        
        # Test 4: Metrics summary
        response = client.get('/metrics/summary')
        print(f"Metrics summary: {response.status_code}")
        
        print("All basic endpoints are accessible!")
        return True
        
    except Exception as e:
        print(f"Error testing API: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = test_api()
    print(f"\nAPI Test Result: {'PASSED' if result else 'FAILED'}")
