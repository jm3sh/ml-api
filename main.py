from flask import Flask, request, jsonify
import customer_insights
import sales_forecast
import product_demand
import anomaly_detection
import auto_forecast_runner
import full_forecast

app = Flask(__name__)

# ----------------------------
# Customer Insights
# ----------------------------
@app.route('/customer_insights', methods=['POST'])
def customer_insights_route():
    try:
        data = request.get_json(force=True)
        result = customer_insights.run(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ----------------------------
# Sales Forecast
# ----------------------------
@app.route('/sales_forecast', methods=['POST'])
def sales_forecast_route():
    try:
        data = request.get_json(force=True)
        result = sales_forecast.run(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ----------------------------
# Product Demand
# ----------------------------
@app.route('/product_demand', methods=['POST'])
def product_demand_route():
    try:
        data = request.get_json(force=True)
        result = product_demand.run(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ----------------------------
# Anomaly Detection
# ----------------------------
@app.route('/anomaly_detection', methods=['POST'])
def anomaly_detection_route():
    try:
        data = request.get_json(force=True)
        result = anomaly_detection.run(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ----------------------------
# Full Forecast (still DB-based)
# ----------------------------
@app.route('/run_full_forecast', methods=['GET'])
def run_full_forecast():
    try:
        forecast = full_forecast.generate_forecast(days_back=60, days_ahead=7)
        if not forecast or 'predictions' not in forecast:
            return jsonify({'success': False, 'error': 'Forecast generation failed.'})

        inserted = auto_forecast_runner.insert_forecast_to_db(
            forecast['predictions'],
            r2=forecast.get('r2_score'),
            mae=forecast.get('mae')
        )

        return jsonify({
            'success': True,
            'inserted': inserted,
            'forecast': forecast
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ----------------------------
# Health Check
# ----------------------------
@app.route('/', methods=['GET'])
def home():
    return "ML API is running!"

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
