# DocuBot: A 100% Offline RAG Chatbot

This project is a complete, 100% offline, GPU-accelerated RAG (Retrieval-Augmented Generation) chatbot. It is designed to be a self-contained, portable application using Docker, allowing you to run a powerful AI assistant on your own documentation (e.g., C++ and Python docs) on any machine.

This repository contains two main workflows:

1.  **Local Development:** For building the database, testing changes, and running the app locally without Docker.

2.  **Docker Deployment:** For packaging the entire application (including the database) into a portable Docker image that can be run anywhere with a single command.

## 💻 Local Development & Data Building (The "Normal" Way)

Use this method on your main development machine to build your database from scratch or test locally.

### Step 1: Install Local Tools

```bash
# Install Python tools
sudo apt update
sudo apt install -y python3-pip python3-venv wget unzip

# Install Ollama
curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
````

### Step 2: Set Up Project

```bash
# Clone this repository
git clone <your-repo-url>
cd DocuBot-Repo

# Create your Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all local dependencies
pip install -r requirements_local.txt
```

### Step 3: Build Your Database (One-Time Cost)

This is the most important part of the "Phase A" build.

1.  **Download HTML Docs:**
    Create the `html_docs/` folder, then run these commands to download the documentation archives.

    ```bash
    # Download Python 3.12 Docs
    wget [https://docs.python.org/3.12/archives/python-3.12.4-docs-html.zip](https://docs.python.org/3.12/archives/python-3.12.4-docs-html.zip) -O html_docs/python.zip

    # Download C++ (cppreference) Docs
    wget [https://github.com/PeterFeicht/cppreference-doc/releases/download/v20220730/html-book-20220730.zip](https://github.com/PeterFeicht/cppreference-doc/releases/download/v20220730/html-book-20220730.zip) -O html_docs/cpp.zip

    # Unzip them into subfolders
    unzip html_docs/python.zip -d html_docs/python
    unzip html_docs/cpp.zip -d html_docs/cpp
    ```

2.  **Convert HTML to TXT:**
    Run the `convert.py` script. This reads from `html_docs/` and writes to `docs/`.

    ```bash
    python3 convert.py
    ```

3.  **Start Ollama (Local):**
    Make sure your local Ollama server is running (it should start on boot, or run `ollama serve`).

4.  **Pull Models (Local):**

    ```bash
    ollama pull llama3.2:1b
    ollama pull nomic-embed-text
    ```

5.  **Run Ingestion:**
    Run the *local* ingestion script. This connects to `localhost:11434` and builds your `chroma_db/` folder.

    ```bash
    python3 ingest_local.py
    ```

    This will take a long time, but it's a one-time process.

### Step 4: Test Your App (Local)

You can now run your app *without* Docker to test changes quickly.

1.  Make sure your local `ollama serve` is running.

2.  Run the *local* main script:

    ```bash
    python3 main_local.py
    ```

    Your server is now running at `http://localhost:8000`.

## 📦 Building Your Production Docker Image (Phase A)

Once you have built your `chroma_db/` folder locally (using the steps above), you can "bake" it into a portable Docker image to be published.

1.  **Log in to Docker Hub:**

    ```bash
    docker login
    ```

    (Enter your Docker Hub username and password).

2.  **Build your "monolithic" image:**
    *(This will copy your local `chroma_db` folder into the image)*

    ```bash
    # Make sure to replace 'yousefabdelwahab' with your username
    docker build -t yousefabdelwahab/docubot-app:latest .
    ```

3.  **Push your image to Docker Hub:**
    *(This will be a large, slow upload)*

    ```bash
    docker push yousefabdelwahab/docubot-app:latest
    ```

    Your app is now published and ready for the "Easy Deployment" (Phase B) workflow.

## 🚀 Deployment (The Easy "Portable" Way)

This is the main "production" workflow. Use this to run your chatbot on any new machine that has Docker and NVIDIA drivers installed.

**This method uses the pre-built Docker image** (which you either built in the previous step or can be pulled if it's public). This image already contains the application and the `chroma_db` database.

### Step-by-Step on a New Machine

1.  **Install Docker & NVIDIA Drivers:**

      * Install Docker: `curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh`
      * Install the NVIDIA Container Toolkit:
        ```bash
        curl -fsSL [https://nvidia.github.io/libnvidia-container/gpgkey](https://nvidia.github.io/libnvidia-container/gpgkey) | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
          && curl -s -L [https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list](https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list) | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        ```
      * Restart the Docker service: `sudo systemctl restart docker`

2.  **Create Your Project Folder:**

    ```bash
    mkdir MyBotRunner
    cd MyBotRunner
    ```

3.  **Add Your Project Files:**

      * Create the `docker-compose.yml` file in this folder (copy it from this repository).
      * **Note:** The `docker-compose.yml` file is configured to use a **named volume** (`ollama_data`) to store models.

    Your folder should look like this (if using the named volume method):

    ```
    MyBotRunner/
    └── docker-compose.yml
    ```

4.  **Run It\!**

    ```bash
    docker compose up -d
    ```

    Docker will:

    1.  **Pull** your `yousefabdelwahab/docubot-app:latest` image (which has the database inside).
    2.  **Pull** the `ollama/ollama` image and create the `ollama_data` volume.
    3.  Start both containers.

5.  **Pull AI Models (One-Time-Only per Machine)**
    Your containers are running, but the `ollama_data` volume is empty. You must tell Ollama to download the models *once*.

    ```bash
    # Pull the chat model
    docker compose exec ollama ollama pull llama3.2:1b

    # Pull the embedding model
    docker compose exec ollama ollama pull nomic-embed-text
    ```

    Your chatbot is now fully operational at `http://localhost:8000`. You will never have to run Step 5 on this machine again.
