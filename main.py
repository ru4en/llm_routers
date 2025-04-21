from llm_routers import Router
import sys

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        print("Usage: python main.py <query>")
        sys.exit(1)
    query = " ".join(args)
    



if __name__ == "__main__":
    main()
