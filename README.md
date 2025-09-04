Master Degree Thesis

## Run solution
- Ensure having the following installed:
    - [Docker](https://www.docker.com/)
    - [VS Code](https://code.visualstudio.com/)
        - [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension

- It is sufficient to use _Dev Containers_ extension's 'Dev Containers: Rebuild and Reopen in Container' command.

## Project setup
- [uv](https://docs.astral.sh/uv/) for package management

## Shared resources
- [Google Drive](https://drive.google.com/drive/folders/1zr0JyMETI44G2ikBb93U3lLXUdiOqqnZ?usp=sharing) for fine-tuned models

<!-- TODO: Mention about settings required to be made for using Gmail provider

Steps
1.	Create / select a project → APIs & Services ▸ Enabled APIs & services ▸ + ENABLE APIs → turn on Gmail API.
2.	OAuth consent screen → External → add your own Google account as a Test user.
3.	Create credentials → OAuth client ID ▸ Desktop app.
•	Download the client_secret_<…>.json file; place it next to your code (or mount it into Docker) and keep it private.
4.	First run locally: the script below pops a browser asking you to log in and grant the single scope https://www.googleapis.com/auth/gmail.send.
Remarks: For 4 run 'Init gcloud CLI' task
Google returns a refresh token which ends up in token.json; afterwards the script renews access tokens silently.

Resources:
1. (Create Gmail API App in the Google Developer Console)](https://www.youtube.com/watch?app=desktop&v=1Ua0Eplg75M)
2. ChatGPT
 -->