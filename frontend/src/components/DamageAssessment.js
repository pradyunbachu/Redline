import React, { useState } from "react";
import axios from "axios";
import "./DamageAssessment.css";
import ChatAgent from "./ChatAgent";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:5001";

const DamageAssessment = () => {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [vehicleInfo, setVehicleInfo] = useState({
    make: "",
    model: "",
    year: "",
    mileage: "",
  });

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
      setResults(null);
      setError(null);
    }
  };

  const handleVehicleInfoChange = (e) => {
    const { name, value } = e.target;
    setVehicleInfo((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!image) {
      setError("Please select an image first");
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append("image", image);

      // Add vehicle info if provided
      const info = {};
      if (vehicleInfo.make) info.make = vehicleInfo.make;
      if (vehicleInfo.model) info.model = vehicleInfo.model;
      if (vehicleInfo.year) info.year = parseInt(vehicleInfo.year);
      if (vehicleInfo.mileage) info.mileage = parseInt(vehicleInfo.mileage);

      if (Object.keys(info).length > 0) {
        formData.append("vehicle_info", JSON.stringify(info));
      }

      const response = await axios.post(`${API_URL}/api/assess`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      if (response.data.success) {
        setResults(response.data);
      } else {
        setError(response.data.error || "Failed to assess damage");
      }
    } catch (err) {
      console.error("Error:", err);
      setError(
        err.response?.data?.error ||
          err.message ||
          "An error occurred while processing the image"
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImage(null);
    setPreview(null);
    setResults(null);
    setError(null);
    setVehicleInfo({
      make: "",
      model: "",
      year: "",
      mileage: "",
    });
  };

  return (
    <div className="damage-assessment">
      <div className="container">
        <div className="upload-section">
          <div className="upload-card">
            <h2>Upload Vehicle Image</h2>

            <form onSubmit={handleSubmit}>
              <div className="image-upload">
                <label htmlFor="image-upload" className="upload-label">
                  {preview ? (
                    <img
                      src={preview}
                      alt="Preview"
                      className="preview-image"
                    />
                  ) : (
                    <div className="upload-placeholder">
                      <svg
                        width="64"
                        height="64"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                      </svg>
                      <p>Click to upload or drag and drop</p>
                      <p className="hint">PNG, JPG, JPEG up to 10MB</p>
                    </div>
                  )}
                </label>
                <input
                  id="image-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleImageChange}
                  className="file-input"
                />
              </div>

              <div className="vehicle-info">
                <h3>Vehicle Information (Optional)</h3>
                <div className="info-grid">
                  <div className="info-field">
                    <label>Make</label>
                    <input
                      type="text"
                      name="make"
                      value={vehicleInfo.make}
                      onChange={handleVehicleInfoChange}
                      placeholder="e.g., Toyota"
                    />
                  </div>
                  <div className="info-field">
                    <label>Model</label>
                    <input
                      type="text"
                      name="model"
                      value={vehicleInfo.model}
                      onChange={handleVehicleInfoChange}
                      placeholder="e.g., Camry"
                    />
                  </div>
                  <div className="info-field">
                    <label>Year</label>
                    <input
                      type="number"
                      name="year"
                      value={vehicleInfo.year}
                      onChange={handleVehicleInfoChange}
                      placeholder="e.g., 2020"
                      min="1900"
                      max="2030"
                    />
                  </div>
                  <div className="info-field">
                    <label>Mileage</label>
                    <input
                      type="number"
                      name="mileage"
                      value={vehicleInfo.mileage}
                      onChange={handleVehicleInfoChange}
                      placeholder="e.g., 30000"
                      min="0"
                    />
                  </div>
                </div>
              </div>

              <div className="button-group">
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={!image || loading}>
                  {loading ? (
                    <>
                      <span className="spinner"></span>
                      Analyzing...
                    </>
                  ) : (
                    "Assess Damage"
                  )}
                </button>
                {(preview || results) && (
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={handleReset}>
                    Reset
                  </button>
                )}
              </div>
            </form>
          </div>
        </div>

        {error && (
          <div className="error-card">
            <h3>⚠️ Error</h3>
            <p>{error}</p>
          </div>
        )}

        {results && (
          <div className="results-section">
            <div className="results-card">
              <div className="results-header">
                <h2>Assessment Results</h2>
                <div className="total-cost">
                  <span className="cost-label">Total Estimated Cost</span>
                  <span className="cost-value">
                    ${results.total_estimated_cost.toLocaleString()}
                  </span>
                </div>
              </div>

              <div className="summary">
                <div className="summary-item">
                  <span className="summary-label">Damage Instances Found</span>
                  <span className="summary-value">{results.num_damages}</span>
                </div>
              </div>

              {results.valuation_info && results.vehicle_multiplier !== 1.0 && (
                <div className="valuation-section">
                  <h3>💰 Vehicle-Based Cost Adjustment</h3>
                  <div className="valuation-card">
                    <div className="valuation-row">
                      <span className="valuation-label">Vehicle Class:</span>
                      <span className="valuation-value">
                        {results.valuation_info.luxury_tier || "Standard"}
                      </span>
                    </div>
                    {results.valuation_info.make && (
                      <div className="valuation-row">
                        <span className="valuation-label">Make/Model:</span>
                        <span className="valuation-value">
                          {results.valuation_info.make.charAt(0).toUpperCase() +
                            results.valuation_info.make.slice(1)}
                          {results.valuation_info.model &&
                            ` ${
                              results.valuation_info.model
                                .charAt(0)
                                .toUpperCase() +
                              results.valuation_info.model.slice(1)
                            }`}
                        </span>
                      </div>
                    )}
                    {results.valuation_info.year && (
                      <div className="valuation-row">
                        <span className="valuation-label">
                          Year Adjustment:
                        </span>
                        <span className="valuation-value">
                          {results.valuation_info.year} (
                          {results.valuation_info.year_adjustment > 1
                            ? "+"
                            : ""}
                          {(
                            (results.valuation_info.year_adjustment - 1) *
                            100
                          ).toFixed(0)}
                          %)
                        </span>
                      </div>
                    )}
                    {results.valuation_info.mileage && (
                      <div className="valuation-row">
                        <span className="valuation-label">
                          Mileage Adjustment:
                        </span>
                        <span className="valuation-value">
                          {results.valuation_info.mileage.toLocaleString()} mi (
                          {results.valuation_info.mileage_adjustment > 1
                            ? "+"
                            : ""}
                          {(
                            (results.valuation_info.mileage_adjustment - 1) *
                            100
                          ).toFixed(0)}
                          %)
                        </span>
                      </div>
                    )}
                    <div className="valuation-multiplier">
                      <span className="multiplier-label">
                        Total Cost Multiplier:
                      </span>
                      <span
                        className={`multiplier-value ${
                          results.vehicle_multiplier > 1
                            ? "increase"
                            : results.vehicle_multiplier < 1
                            ? "decrease"
                            : ""
                        }`}>
                        {results.vehicle_multiplier > 1
                          ? "↑"
                          : results.vehicle_multiplier < 1
                          ? "↓"
                          : ""}{" "}
                        {results.vehicle_multiplier.toFixed(2)}x
                      </span>
                      <span className="multiplier-explanation">
                        {results.vehicle_multiplier > 1
                          ? "Higher costs due to luxury vehicle, newer year, or low mileage"
                          : results.vehicle_multiplier < 1
                          ? "Lower costs due to economy vehicle, older year, or high mileage"
                          : "Standard pricing applied"}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {results.visualization && (
                <div className="visualization-section">
                  <h3>Damage Detection Visualization</h3>
                  <div className="visualization-container">
                    <img
                      src={results.visualization}
                      alt="Damage assessment with bounding boxes"
                      className="visualization-image"
                    />
                    <div className="visualization-legend">
                      <div className="legend-item">
                        <span className="legend-color minor"></span>
                        <span>Minor</span>
                      </div>
                      <div className="legend-item">
                        <span className="legend-color moderate"></span>
                        <span>Moderate</span>
                      </div>
                      <div className="legend-item">
                        <span className="legend-color severe"></span>
                        <span>Severe</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {results.damage_instances &&
                results.damage_instances.length > 0 && (
                  <div className="damage-list">
                    <h3>Damage Details</h3>
                    {results.damage_instances.map((damage, index) => (
                      <div key={index} className="damage-item">
                        <div className="damage-header">
                          <span className="damage-number">#{index + 1}</span>
                          <span
                            className={`severity-badge severity-${damage.severity_class}`}>
                            {damage.severity_class.toUpperCase()}
                          </span>
                        </div>

                        <div className="damage-info">
                          <div className="info-row">
                            <span className="info-label">Type:</span>
                            <span className="info-value">
                              {damage.damage_class}
                            </span>
                          </div>
                          <div className="info-row">
                            <span className="info-label">Part:</span>
                            <span className="info-value">
                              {damage.part_name.replace("_", " ")}
                            </span>
                          </div>
                          <div className="info-row">
                            <span className="info-label">Confidence:</span>
                            <span className="info-value">
                              {(damage.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="info-row">
                            <span className="info-label">Severity Score:</span>
                            <span className="info-value">
                              {damage.severity_score}
                            </span>
                          </div>
                        </div>

                        <div className="cost-breakdown">
                          {damage.cost_estimate.rule_breakdown.part_cost >
                            0 && (
                            <div className="cost-item">
                              <span className="cost-label">Part Cost:</span>
                              <span className="cost-amount">
                                $
                                {damage.cost_estimate.rule_breakdown.part_cost.toLocaleString(
                                  undefined,
                                  {
                                    minimumFractionDigits: 2,
                                    maximumFractionDigits: 2,
                                  }
                                )}
                              </span>
                            </div>
                          )}
                          <div className="cost-item">
                            <span className="cost-label">Labor:</span>
                            <span className="cost-amount">
                              $
                              {damage.cost_estimate.rule_breakdown.labor_cost.toLocaleString(
                                undefined,
                                {
                                  minimumFractionDigits: 2,
                                  maximumFractionDigits: 2,
                                }
                              )}
                            </span>
                          </div>
                          {damage.cost_estimate.rule_breakdown.paint_cost >
                            0 && (
                            <div className="cost-item">
                              <span className="cost-label">
                                Paint & Materials:
                              </span>
                              <span className="cost-amount">
                                $
                                {damage.cost_estimate.rule_breakdown.paint_cost.toLocaleString(
                                  undefined,
                                  {
                                    minimumFractionDigits: 2,
                                    maximumFractionDigits: 2,
                                  }
                                )}
                              </span>
                            </div>
                          )}
                          {damage.cost_estimate.rule_breakdown.shop_supplies >
                            0 && (
                            <div className="cost-item">
                              <span className="cost-label">Shop Supplies:</span>
                              <span className="cost-amount">
                                $
                                {damage.cost_estimate.rule_breakdown.shop_supplies.toLocaleString(
                                  undefined,
                                  {
                                    minimumFractionDigits: 2,
                                    maximumFractionDigits: 2,
                                  }
                                )}
                              </span>
                            </div>
                          )}
                          {damage.cost_estimate.rule_breakdown.disposal_fee >
                            0 && (
                            <div className="cost-item">
                              <span className="cost-label">Disposal Fee:</span>
                              <span className="cost-amount">
                                $
                                {damage.cost_estimate.rule_breakdown.disposal_fee.toLocaleString(
                                  undefined,
                                  {
                                    minimumFractionDigits: 2,
                                    maximumFractionDigits: 2,
                                  }
                                )}
                              </span>
                            </div>
                          )}
                          {damage.cost_estimate.rule_breakdown.additional_fees >
                            0 && (
                            <div className="cost-item">
                              <span className="cost-label">
                                Additional Fees:
                              </span>
                              <span className="cost-amount">
                                $
                                {damage.cost_estimate.rule_breakdown.additional_fees.toLocaleString(
                                  undefined,
                                  {
                                    minimumFractionDigits: 2,
                                    maximumFractionDigits: 2,
                                  }
                                )}
                              </span>
                            </div>
                          )}
                          <div className="cost-item total">
                            <span className="cost-label">Total:</span>
                            <span className="cost-amount">
                              $
                              {damage.cost_estimate.final_cost.toLocaleString(
                                undefined,
                                {
                                  minimumFractionDigits: 2,
                                  maximumFractionDigits: 2,
                                }
                              )}
                            </span>
                          </div>
                          <div className="action-badge">
                            {damage.cost_estimate.rule_breakdown
                              .replace_or_repair === "replace"
                              ? "🔧 Replace"
                              : "🛠️ Repair"}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

              {results && (
                <>
                  <div className="chat-section">
                    <h3>Ask About Your Estimate</h3>
                    <ChatAgent
                      damageResults={results}
                      originalImage={preview}
                    />
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DamageAssessment;
