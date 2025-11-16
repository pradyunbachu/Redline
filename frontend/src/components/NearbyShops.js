import React, { useState } from "react";
import "./NearbyShops.css";

const NearbyShops = () => {
  const [loading, setLoading] = useState(false);
  const [shops, setShops] = useState(null);
  const [error, setError] = useState(null);
  const [location, setLocation] = useState(null);
  const [radius, setRadius] = useState(10); // Default 10km

  const findNearbyShops = async () => {
    setLoading(true);
    setError(null);
    setShops(null);

    try {
      // Get user's location
      if (!navigator.geolocation) {
        throw new Error("Geolocation is not supported by your browser");
      }

      const position = await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject, {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 0
        });
      });

      const { latitude, longitude } = position.coords;
      setLocation({ latitude, longitude });

      // Call backend API
      const response = await fetch("http://localhost:5001/api/nearby-shops", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          latitude,
          longitude,
          radius: radius * 1000, // Convert km to meters
        }),
      });

      const data = await response.json();

      if (data.success) {
        setShops(data.shops);
      } else {
        throw new Error(data.error || "Failed to find nearby shops");
      }
    } catch (err) {
      console.error("Error finding nearby shops:", err);
      if (err.code === 1) {
        setError("Location access denied. Please allow location access to find nearby shops.");
      } else if (err.code === 2) {
        setError("Location unavailable. Please check your device's location settings.");
      } else if (err.code === 3) {
        setError("Location request timed out. Please try again.");
      } else {
        setError(err.message || "Failed to find nearby shops. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  const getDirections = (shop) => {
    const url = `https://www.google.com/maps/dir/?api=1&destination=${shop.latitude},${shop.longitude}`;
    window.open(url, "_blank");
  };

  const getMapUrl = () => {
    if (!location || !shops || shops.length === 0) return null;
    
    // Create Google Maps static image URL with markers
    const center = `${location.latitude},${location.longitude}`;
    const markers = shops.map((shop, index) => {
      const color = index === 0 ? 'red' : 'blue';
      return `markers=color:${color}|label:${index + 1}|${shop.latitude},${shop.longitude}`;
    }).join('&');
    
    // Add user location marker
    const userMarker = `markers=color:green|label:You|${location.latitude},${location.longitude}`;
    
    return `https://maps.googleapis.com/maps/api/staticmap?center=${center}&zoom=13&size=600x400&${userMarker}&${markers}&key=${process.env.REACT_APP_GOOGLE_MAPS_API_KEY || ''}`;
  };

  const getEmbedMapUrl = () => {
    if (!location || !shops || shops.length === 0) return null;
    
    // Create Google Maps embed URL
    const center = `${location.latitude},${location.longitude}`;
    const places = shops.map(shop => `${shop.latitude},${shop.longitude}`).join('/');
    
    return `https://www.google.com/maps/embed/v1/view?key=${process.env.REACT_APP_GOOGLE_MAPS_API_KEY || ''}&center=${center}&zoom=13`;
  };

  return (
    <div className="nearby-shops">
      <div className="nearby-shops-header">
        <h3>🔧 Find Nearby Auto Shops</h3>
        <p>Get directions to the nearest repair shops</p>
      </div>

      <div className="radius-selector">
        <label htmlFor="radius-select">Search Radius:</label>
        <select
          id="radius-select"
          value={radius}
          onChange={(e) => setRadius(Number(e.target.value))}
          disabled={loading}
        >
          <option value={5}>5 km</option>
          <option value={10}>10 km</option>
          <option value={15}>15 km</option>
          <option value={25}>25 km</option>
          <option value={50}>50 km</option>
        </select>
      </div>

      <button
        className="find-shops-btn"
        onClick={findNearbyShops}
        disabled={loading}
      >
        {loading ? "Searching..." : "Find Nearby Shops"}
      </button>

      {error && (
        <div className="shops-error">
          <p>❌ {error}</p>
        </div>
      )}

      {shops && shops.length > 0 && (
        <div className="shops-results">
          <h4>Found {shops.length} nearby shop{shops.length !== 1 ? "s" : ""}</h4>
          
          {/* Google Maps View */}
          <div className="map-container">
            {process.env.REACT_APP_GOOGLE_MAPS_API_KEY ? (
              <iframe
                width="100%"
                height="400"
                style={{ border: 0, borderRadius: "8px" }}
                loading="lazy"
                allowFullScreen
                referrerPolicy="no-referrer-when-downgrade"
                src={getEmbedMapUrl()}
                title="Nearby Auto Shops Map"
              ></iframe>
            ) : (
              <div className="map-placeholder">
                <p>📍 Map view requires Google Maps API key</p>
                <p className="map-instructions">
                  Add <code>REACT_APP_GOOGLE_MAPS_API_KEY</code> to your .env file to see the map.
                  <br />
                  For now, use "Get Directions" buttons to open Google Maps.
                </p>
                {getMapUrl() && (
                  <img 
                    src={getMapUrl()} 
                    alt="Nearby shops map" 
                    style={{ maxWidth: "100%", borderRadius: "8px" }}
                  />
                )}
              </div>
            )}
          </div>

          <div className="shops-list">
            {shops.map((shop, index) => (
              <div key={index} className="shop-card">
                <div className="shop-info">
                  <h5>{shop.name}</h5>
                  <p className="shop-address">{shop.address}</p>
                  {shop.rating > 0 && (
                    <div className="shop-rating">
                      <span className="rating-stars">
                        {"⭐".repeat(Math.round(shop.rating))}
                      </span>
                      <span className="rating-value">
                        {shop.rating.toFixed(1)} ({shop.rating_count} reviews)
                      </span>
                    </div>
                  )}
                  {shop.open_now !== null && (
                    <p className="shop-status">
                      {shop.open_now ? "🟢 Open now" : "🔴 Closed"}
                    </p>
                  )}
                </div>
                <button
                  className="directions-btn"
                  onClick={() => getDirections(shop)}
                >
                  Get Directions
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {shops && shops.length === 0 && (
        <div className="shops-empty">
          <p>No auto shops found within {radius} km. Try expanding your search radius above.</p>
          <button
            className="retry-btn"
            onClick={() => {
              setRadius(Math.min(radius + 10, 50));
              findNearbyShops();
            }}
          >
            Search with Larger Radius ({Math.min(radius + 10, 50)} km)
          </button>
        </div>
      )}
    </div>
  );
};

export default NearbyShops;

