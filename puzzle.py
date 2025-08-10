import requests
import asyncio

PARALLEL_WORLD_MAP = {
    "Delhi": "Gateway of India",
    "Mumbai": "India Gate",
    # ... add all mappings here ...
    "Los Angeles": "Buckingham Palace"
}

FLIGHT_ENDPOINTS = {
    "Gateway of India": "getFirstCityFlightNumber",
    "Taj Mahal": "getSecondCityFlightNumber",
    # ... add all endpoints here ...
}

FALLBACK_ENDPOINT = "getFifthCityFlightNumber"

async def fetch_flight_number_async() -> str:
    def sync_logic():
        STEP1_URL = "https://register.hackrx.in/submissions/myFavouriteCity"
        resp = requests.get(STEP1_URL, timeout=10)
        resp.raise_for_status()
        city = resp.json().get("data", {}).get("city", "").strip()
        if not city:
            raise RuntimeError("Failed to get favorite city from API")
        landmark = PARALLEL_WORLD_MAP.get(city)
        if not landmark:
            raise RuntimeError(f"City '{city}' not found in parallel world map")
        
        endpoint = FLIGHT_ENDPOINTS.get(landmark, FALLBACK_ENDPOINT)
        flight_url = f"https://register.hackrx.in/teams/public/flights/{endpoint}"
        flight_resp = requests.get(flight_url, timeout=10)
        flight_resp.raise_for_status()
        
        flight_number = flight_resp.json().get("data", {}).get("flightNumber")
        if not flight_number:
            raise RuntimeError("Failed to get flight number from response")
        
        return f"Your flight number is {flight_number}."
    
    return await asyncio.to_thread(sync_logic)
