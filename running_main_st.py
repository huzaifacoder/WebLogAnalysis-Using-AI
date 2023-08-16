import os


def streamlit_cmd():
    # Change directory to the location of main.py
    script_dir = os.path.dirname(os.path.realpath("main.py"))
    os.chdir(script_dir)

    # Run the Streamlit app using streamlit run main.py with --no-reload option
    os.system("streamlit run main.py")

if __name__ == "__main__":
    streamlit_cmd()