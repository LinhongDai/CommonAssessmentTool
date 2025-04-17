Team TicTech 

Project -- Feature Development Backend: Create CRUD API's for Client

User Story

As a user of the backend API's, I want to call API's that can retrieve, update, and delete information of clients who have already registered with the CaseManagment service so that I more efficiently help previous clients make better decisions on how to be gainfully employed.

Acceptance Criteria
- Provide REST API endpoints so that the Frontend can use them to get information on an existing client.
- Document how to use the REST API
- Choose and create a database to hold client information
- Add tests


This will contain the model used for the project that based on the input information will give the social workers the clients baseline level of success and what their success will be after certain interventions.

The model works off of dummy data of several combinations of clients alongside the interventions chosen for them as well as their success rate at finding a job afterward. The model will be updated by the case workers by inputing new data for clients with their updated outcome information, and it can be updated on a daily, weekly, or monthly basis.

This also has an API file to interact with the front end, and logic in order to process the interventions coming from the front end. This includes functions to clean data, create a matrix of all possible combinations in order to get the ones with the highest increase of success, and output the results in a way the front end can interact with.

-------------------------How to Use-------------------------
1. In the virtual environment you've created for this project, install all dependencies in requirements.txt (pip install -r requirements.txt)

2. Run the app (uvicorn app.main:app --reload)

3. Load data into database (python initialize_data.py)

4. Go to SwaggerUI (http://127.0.0.1:8000/docs)

4. Log in as admin (username: admin password: admin123)

5. Click on each endpoint to use
-Create User (Only users in admin role can create new users. The role field needs to be either "admin" or "case_worker")

-Get clients (Display all the clients that are in the database)

-Get client (Allow authorized users to search for a client by id. If the id is not in database, an error message will show.)

-Update client (Allow authorized users to update a client's basic info by inputting in client_id and providing updated values.)

-Delete client (Allow authorized users to delete a client by id. If an id is no longer in the database, an error message will show.)

-Get clients by criteria (Allow authorized users to get a list of clients who meet a certain combination of criteria.)

-Get Clients by services (Allow authorized users to get a list of clients who meet a certain combination of service statuses.)

-Get clients services (Allow authorized users to view a client's services' status.)

-Get clients by success rate (Allow authorized users to search for clients whose cases have a success rate beyond a certain number.)

-Get clients by case worker (Allow users to view which clients are assigned to a specific case worker.)

-Update client services (Allow users to update the service status of a case.)

-Create case assignment (Allow authorized users to create a new case assignment.)



## Running Backend with Docker Compose

### Prerequisites
Before you start, make sure you have:
- Install Docker if you haven't already.

- You can check if they are installed with:
```
docker --version
```

### Build and start the backend service

This will:

- Build the Docker image using the provided Dockerfile

- Start a container based on that image

```bash
docker compose up --build
```

- Start without rebuilding:
```
docker compose up
```


### Stop the service
```bash
docker compose down
```


To automate our development workflow and ensure code quality, we implemented a CI pipeline using GitHub Actions.

### Publicly Available Endpoint
Access the backend application and test the various features of the application using the Swagger from:

```
http://34.222.122.12:8000/docs
```


### GitHub Workflow Setup

Developer C created the GitHub Actions workflow file located at `.github/workflows/ci.yml`. This workflow is configured to trigger on every pull request or direct push to the `main` branch, ensuring all changes are validated before merging.

### CI Jobs

Developer D implemented the following CI jobs:

- **Task 1: Linter/Formatter Check**
  - Tools used: `flake8`, `black`
  - Ensures consistent coding style and helps detect syntax or logic errors early.

- **Task 2: Unit Testing**
  - Tool used: `pytest`
  - Runs all unit tests to validate functionality and prevent regressions.

- **Task 3: Docker Syntax Validation & Image Build**
  - Validates the syntax of Docker-related files and attempts to build the Docker image.

- **Task 4: Container Startup Verification**
  - Executes the built Docker container within the CI pipeline to ensure it can start correctly.

### Definition of Done (DoD)

The pipeline is considered complete when all CI checks (linter, tests, Docker validations) pass successfully on pull request or push to the `main` branch.


### Continuous Deployment (CD) Pipeline

This project includes a Continuous Deployment (CD) pipeline using GitHub Actions. The pipeline is triggered automatically whenever a new GitHub Release is published. The goal is to fully automate the deployment process, eliminating the need for manual intervention.

### Deployment Workflow

The CD workflow consists of the following steps:

- ** 1. Run Tests Before Deployment

Before deploying to the server, the pipeline runs automated tests using GitHub Actions. For example, pytest is used to validate that the backend code passes all unit and integration tests. This step ensures that faulty code is never deployed.
	•	If any test fails, the pipeline halts and deployment does not proceed.
	•	If all tests pass, the deployment continues.

- ** 2. SSH into EC2 Instance Automatically

GitHub Actions uses a private SSH key (stored securely in GitHub Secrets, e.g., FastAPI.pem) to access the target EC2 instance.
	•	This replaces the manual login process and ensures secure, automated access.
	•	The SSH command used is equivalent to:

```bash
 ssh -i FastAPI.pem ec2-user@<your-ec2-ip>
```

	•	No manual input is required during this process.


- ** 3. Pull the Latest Code from GitHub

Once logged into the EC2 instance, the workflow navigates to the project directory and pulls the latest version of the code from the main branch:

```bash
cd CommonAssessmentTool
git pull origin main
```
This ensures that the EC2 instance is running the most recent codebase that corresponds to the latest release.


- ** 4. Activate the Virtual Environment and Restart the FastAPI Application

After pulling the updated code, the workflow activates the Python virtual environment and starts the FastAPI application using Uvicorn:

```bash
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Explanation:
	•	source venv/bin/activate activates the virtual environment to ensure dependencies are available.
	•	uvicorn app.main:app --host 0.0.0.0 --port 8000 launches the FastAPI server, making it accessible via port 8000.

### Summary

This automated CD pipeline allows you to deploy new versions of your backend simply by publishing a GitHub Release. The system will:
	1.	Run tests to ensure code quality
	2.	SSH into your EC2 instance securely
	3.	Pull the latest code from GitHub
	4.	Restart the FastAPI application

No manual SSH commands or deployment scripts are required, significantly reducing operational overhead and the risk of human error.




