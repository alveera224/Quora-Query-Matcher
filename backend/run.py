import uvicorn

if __name__ == "__main__":
    print("Starting Quora Query Matcher API server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 