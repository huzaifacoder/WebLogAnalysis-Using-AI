import os


def streamlit_cmd():
    # Change directory to the location of Reference.py
    script_dir = os.path.dirname(os.path.realpath("pages/Reference.py"))
    os.chdir(script_dir)

    # Run the Streamlit app using streamlit run Reference.py with --no-reload option
    os.system("streamlit run Reference.py")

if __name__ == "__main__":
    streamlit_cmd()