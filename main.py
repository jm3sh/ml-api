from flask import Flask, request, jsonify
import customer_insights
import sales_forecast
import product_demand
import anomaly_detection

app = Flask(__name__)

@app.route('/customer_insights', methods=['POST'])
def customer_insights_route():
    data = request.get_json()
    result = customer_insights.run(data)
    return jsonify(result)

@app.route('/sales_forecast', methods=['POST'])
def sales_forecast_route():
    data = request.get_json()
    result = sales_forecast.run(data)
    return jsonify(result)

@app.route('/product_demand', methods=['POST'])
def product_demand_route():
    data = request.get_json()
    result = product_demand.run(data)
    return jsonify(result)

@app.route('/anomaly_detection', methods=['POST'])
def anomaly_detection_route():
    data = request.get_json()
    result = anomaly_detection.run(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run()