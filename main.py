import uvicorn
from os import getenv


if __name__ == "__main":
    port = int(getenv("PORT",3000))
    uvicorn.run("api:api",host="0.0.0.0",port=port,reload=True)