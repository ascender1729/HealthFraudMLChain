<table>
  <tr>
    <th>Development Area</th>
    <th>Tools/Technologies</th>
    <th>Description</th>
  </tr>
  <tr>
    <td rowspan="2"><img src="https://img.shields.io/badge/Frontend_Development-000?style=for-the-badge" alt="Frontend Development"></td>
    <td><img src="https://img.shields.io/badge/HTML%2FCSS-blue?style=for-the-badge" alt="HTML/CSS"></td>
    <td>For structuring and styling web pages, ensuring an intuitive and responsive user interface.</td>
  </tr>
  <tr>
    <td><img src="https://img.shields.io/badge/Jinja2-blue?style=for-the-badge" alt="Jinja2"></td>
    <td>A templating engine for Python, used for generating HTML pages with dynamic content.</td>
  </tr>
  <tr>
    <td rowspan="2"><img src="https://img.shields.io/badge/Backend_Development-000?style=for-the-badge" alt="Backend Development"></td>
    <td><img src="https://img.shields.io/badge/-Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"></td>
    <td>A lightweight WSGI web framework for serving the web application.</td>
  </tr>
  <tr>
    <td><img src="https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></td>
    <td>The core programming language, used across backend development and data processing tasks.</td>
<tr>
  <td rowspan="3"><img src="https://img.shields.io/badge/Machine_Learning-000?style=for-the-badge" alt="Machine Learning"></td>
  <td><img src="https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"></td>
  <td>Essential for data manipulation and analysis, enabling efficient handling of datasets.</td>
</tr>
<tr>
  <td><img src="https://img.shields.io/badge/-Scikit_Image-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"></td>
  <td>Used for developing predictive models to identify fraudulent activities.</td>
</tr>
<tr>
  <td><img src="https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"></td>
  <td>Supports high-level mathematical functions and multi-dimensional arrays.</td>
</tr>

<tr>
  <td rowspan="2"><img src="https://img.shields.io/badge/Blockchain_Integration-000?style=for-the-badge" alt="Blockchain Integration"></td>
  <td><img src="https://img.shields.io/badge/Blockchain_Technology-blue?style=for-the-badge" alt="Blockchain Technology"></td>
  <td>Utilized for creating immutable data records, enhancing data security and integrity.</td>
</tr>
<tr>
  <td><img src="https://img.shields.io/badge/-ECIES-4A4A55?style=for-the-badge" alt="ECIES"></td>
  <td>For secure data encryption and decryption, and generating Ethereum-compatible keys.</td>
</tr>

<tr>
  <td rowspan="2"><img src="https://img.shields.io/badge/Cryptography_and_Security-000?style=for-the-badge" alt="Cryptography and Security"></td>
  <td><img src="https://img.shields.io/badge/Hashlib-blue?style=for-the-badge" alt="Hashlib"></td>
  <td>Implements secure hash and message digest algorithms, vital for data integrity checks.</td>
</tr>
<tr>
  <td><img src="https://img.shields.io/badge/ECIES-blue?style=for-the-badge" alt="ECIES"></td>
  <td>Enhances data security through Elliptic Curve Cryptography.</td>
</tr>

<tr>
  <td rowspan="2"><img src="https://img.shields.io/badge/Additional_Tools_and_Libraries-000?style=for-the-badge" alt="Additional Tools and Libraries"></td>
  <td><img src="https://img.shields.io/badge/CSV_JSON-blue?style=for-the-badge" alt="CSV & JSON"></td>
  <td>For handling data in CSV and JSON formats.</td>
</tr>
<tr>
  <td><img src="https://img.shields.io/badge/OS_Warnings-blue?style=for-the-badge" alt="OS & Warnings"></td>
  <td>For performing operating system level operations and managing warnings respectively.</td>
</tr>

<tr>
  <td rowspan="2"><img src="https://img.shields.io/badge/Development_Tools-000?style=for-the-badge" alt="Development Tools"></td>
  <td><img src="https://img.shields.io/badge/-Git-F05032?style=for-the-badge&logo=git&logoColor=white" alt="Git"></td>
  <td>Empowers source code management and collaborative development.</td>
</tr>
<tr>
  <td><img src="https://img.shields.io/badge/-GitHub-222222?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></td>
  <td>Hosts the project repository, providing a platform for version control and collaboration.</td>
</tr>
</table>


## HealthFraudMLChain Setup Guide

### Installation & Setup

1. **Clone the Repository**:
   Open PowerShell and navigate to the directory where you want to clone the repository.
   ```powershell
   git clone https://github.com/ascender1729/HealthFraudMLChain.git
   ```

2. **Navigate to the Project Directory**:
   ```powershell
   cd .\HealthFraudMLChain\code\
   ```

3. **Create and Activate the Virtual Environment**:
   This step is important to ensure that the Python packages installed do not interfere with the packages of other Python projects.
   ```powershell
   python -m venv myenv
   .\myenv\Scripts\Activate.ps1
   ```

4. **Install Dependencies**:
   Once the virtual environment is activated, you'll see `(myenv)` before your directory path in the terminal. Now, install the project dependencies.
   ```powershell
   pip install -r requirements.txt
   ```

### Running the Application

1. **Set Flask Environment Variables**:
   Before running the Flask application, you need to set two environment variables. The `FLASK_APP` variable points to your main application file, and the `FLASK_ENV` sets the environment (development/production).
   ```powershell
   $env:FLASK_APP = "main.py"
   $env:FLASK_ENV = "development"
   ```

2. **Start the Flask Application**:
   To run the Flask application, use the `flask run` command. This will start a local server.
   ```powershell
   flask run
   ```
   You should see output indicating the server has started, similar to this:
   ```
   * Serving Flask app 'main.py'
   * Debug mode: off
   * Running on http://127.0.0.1:5000
   ```

3. **Accessing the Application**:
   Open your web browser and go to `http://127.0.0.1:5000` to view and interact with the Flask application.

### Shutting Down

1. **Deactivate the Virtual Environment**:
   When you are finished working with your Flask application, you can deactivate the virtual environment to return to your global Python environment.
   ```powershell
   deactivate
   ```

Remember to deactivate your virtual environment (`deactivate`) before closing PowerShell or navigating away from the project directory.
