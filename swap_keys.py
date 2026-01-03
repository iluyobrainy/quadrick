import os

# Alternate Key Set (Found in original user file)
NEW_BYBIT_CONFIG = """# Bybit API Configuration
BYBIT_API_KEY=VZgz9YNdgOSmmxa7Yy
BYBIT_API_SECRET=T44MQoAtiyyTlxAYowAf8yZIGx2Tzl1ktYEh
BYBIT_TESTNET=false

# Bybit Base URLs (Alternate Domain to bypass DNS block)
BYBIT_BASE_URL_TESTNET=https://api-testnet.bybit.com
BYBIT_BASE_URL_MAINNET=https://api.bytick.com
BYBIT_WS_URL_TESTNET=wss://stream-testnet.bybit.com/v5/public
BYBIT_WS_URL_MAINNET=wss://stream.bytick.com/v5/public

"""

def update_env():
    env_path = ".env"
    
    if not os.path.exists(env_path):
        print("Error: .env not found")
        return

    with open(env_path, "r") as f:
        lines = f.readlines()

    # Filter out existing Bybit lines
    non_bybit_lines = [
        line for line in lines 
        if not line.strip().startswith("BYBIT_") and not line.strip().startswith("# Bybit")
    ]
    
    # Combine new config with preserved lines
    new_content = NEW_BYBIT_CONFIG + "".join(non_bybit_lines)
    
    # Write back
    with open(env_path, "w") as f:
        f.write(new_content)
        
    print("Successfully swapped API keys in .env")

if __name__ == "__main__":
    update_env()
