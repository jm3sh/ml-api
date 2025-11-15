#!/usr/bin/env python3
"""
Product Demand Forecasting using Linear Regression
Predicts future product demand and recommends restocking
"""

import sys, json, numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from collections import defaultdict

def run(data):
    try:
        if not data:
            return {'success': False, 'error': 'No product sales data available'}

        pd = defaultdict(lambda: {'dates': [], 'quantities': [], 'name': '', 'stock': 0})
        for item in data:
            pid = item['product_id']
            pd[pid]['name'] = item['product_name']
            pd[pid]['stock'] = item['current_stock']
            pd[pid]['dates'].append(datetime.strptime(item['date'], '%Y-%m-%d'))
            pd[pid]['quantities'].append(item['quantity_sold'])

        results, high_priority, total_restock = [], 0, 0
        for pid, p in pd.items():
            if len(p['dates']) < 3: continue
            idx = np.argsort(p['dates'])
            dates = [p['dates'][i] for i in idx]
            qty = np.array([p['quant
