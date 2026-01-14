import os
import redis
from urllib.parse import urlparse

redis_url = os.environ.get("REDIS__PUBLIC_URL")
parsed = urlparse(redis_url)
r = redis.Redis(
    host=parsed.hostname,
    port=parsed.port,
    password=parsed.password,
    decode_responses=True
)

# Stream length
print("Stream length:", r.xlen("transactions.stream"))

# Get last 5 entries
entries = r.xrevrange("transactions.stream", count=5)
for e in entries:
    print(e)
