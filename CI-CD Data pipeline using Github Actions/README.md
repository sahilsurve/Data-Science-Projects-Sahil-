# 🚀 CI/CD Data Pipeline with Databricks, GitHub Actions & Power BI

### 📌 Project Description

This project demonstrates how to implement a **CI/CD workflow for Databricks notebooks using GitHub Actions** and Azure Databricks workspaces.
The setup ensures that updates pushed to the Dev environment are automatically tested and deployed to Prod, with results surfaced in Power BI dashboards for end-users.

This pipeline simulates a real-world Data Engineering lifecycle, combining version control, automated deployments, and business reporting in a seamless loop.

<img width="1918" height="1067" alt="Architecture" src="https://github.com/user-attachments/assets/e83dab08-f039-4b55-94dc-1274be721893" />

### 🚀 Features

- **Two Isolated Environments (Dev & Prod)**\
Managed via separate Azure Resource Groups and Databricks Workspaces for proper environment segregation.

- **GitHub Actions for CI/CD**\
CI: Runs on commits to main branch → syncs code into Dev workspace.\
CD: Deploys validated changes into Prod workspace automatically.

- **Catalog & Schema for Storage**\
Notebook outputs are saved as Delta tables in Databricks Catalog ensuring structured storage.

- **Power BI Integration**\
Leveraged Databricks to connect tables directly into Power BI.\
Dashboard refresh automatically reflects any new data pushed from Dev → Prod.

- **End-to-End Automation**
Updating a notebook in Dev, committing & merging → triggers GitHub Action → updates Prod → refreshes Power BI dashboard.

### 🛠 Tech Stack

- Platform: Azure Databricks (Dev & Prod workspaces)
- Languages: Python (PySpark)
- CI/CD: GitHub Actions
- Storage: Delta Lake (Databricks Catalog & Schema)
- Visualization: Power BI

### 💻 How It Works

1. Developer updates a Databricks notebook in Dev workspace.

2. Commit → Push → Merge PR to main branch.

3. GitHub Action (main.yml) triggers:

- Deploys notebook to Dev.

- Runs tests.

- If successful, deploys notebook to Prod workspace.

- Notebook writes output tables into Prod Databricks Catalog/Schema.

4. Power BI dashboard connected to Prod updates with new data.
