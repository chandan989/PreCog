# Databricks Setup Guide for PreCog Project

This guide provides a step-by-step walkthrough for setting up the necessary Databricks resources for the PreCog project, how to create them, and how they connect.

## Phase 1: Foundational Setup

### Step 1: Create/Access Your Databricks Workspace

*   **What it is:** Your primary, isolated Databricks environment. It's where all your assets (notebooks, clusters, data, etc.) will reside. This is the main "instance" you interact with.
*   **How to create/access:**
    *   This is usually done through your cloud provider's console (AWS, Azure, GCP). You'll provision a "Databricks Service" or "Azure Databricks Workspace," for example.
    *   Once provisioned, you'll get a URL to access your workspace.
*   **Connection:** This is your entry point. Everything else happens *within* this workspace.

### Step 2: Create a Cluster

*   **What it is:** A set of computation resources (virtual machines managed by Databricks) that will run your Spark jobs, notebooks, and ML model training.
*   **How to create:**
    1.  Inside your Databricks Workspace, navigate to the **Compute** or **Clusters** section in the left sidebar.
    2.  Click **Create Cluster**.
    3.  Configure it:
        *   **Cluster Name:** A descriptive name, e.g., `precog-poc-cluster`.
        *   **Databricks Runtime Version:** Choose a recent, stable version that includes Spark, Python, MLflow, etc. (e.g., a version with Scala 2.12, Spark 3.x, and Python 3.x). Check the Databricks documentation for recommended versions.
        *   **Node Types (Worker & Driver):** Select appropriate VM sizes based on your expected workload. For a Proof of Concept (PoC), smaller, general-purpose instances are usually sufficient. You can scale up later if needed.
        *   **Autoscaling:** Enable this if you want the cluster to automatically adjust the number of worker nodes based on the current load. This can be cost-effective.
        *   **Termination:** Set an inactivity timeout (e.g., 120 minutes). The cluster will automatically terminate after this period of inactivity, saving costs.
*   **How it runs/connects:**
    *   Once created, the cluster will start (this involves provisioning the underlying VMs, which can take a few minutes).
    *   You will "attach" your Databricks Notebooks to this cluster to execute code. The code within your notebooks runs on the resources of this cluster.

## Phase 2: Data & Development Setup

### Step 3: Set Up Data Storage & Ingestion Point

*   **What it is:** A location to store your raw, intermediate, and processed data. This will typically be cloud storage (e.g., AWS S3, Azure Data Lake Storage Gen2, Google Cloud Storage) that Databricks can access, or the Databricks File System (DBFS).
*   **How to set up:**
    1.  **Cloud Storage (Recommended for production and flexibility):**
        *   Create a storage container or bucket in your chosen cloud provider's platform.
        *   **Mount** this cloud storage to your Databricks Workspace. This makes the cloud storage location accessible within Databricks as if it were a local directory (e.g., `/mnt/precog_data`). Refer to the Databricks documentation for specific instructions on mounting storage for your cloud provider (AWS, Azure, GCP), as it involves setting up access credentials.
    2.  **DBFS Upload (Simpler for small, PoC datasets):**
        *   For smaller datasets or initial testing, you can directly upload files to DBFS via the Databricks UI. Navigate to the **Data** tab in the sidebar, then select **DBFS**, and use the **Upload** button.
*   **How it runs/connects:**
    *   Your data ingestion scripts (which you'll develop in Databricks Notebooks, conceptually based on your local `src/data_ingestion/` modules) will read data from these source locations.
    *   Processed data will then be written to Delta tables (covered in Step 5), often residing in the same mounted cloud storage.
    *   For your PoC, you can place your sample data from the local `data/` directory into this storage location.

### Step 4: Create Notebooks & Integrate with Git (Databricks Repos)

*   **What they are:** Interactive documents where you write and execute code (Python, SQL, Scala, R), add explanatory markdown text, and visualize results. This is the primary development environment within Databricks.
*   **How to create:**
    1.  In your Databricks Workspace, navigate to the **Workspace** section in the sidebar.
    2.  You can organize your notebooks into folders. Consider creating a folder structure that mirrors your local project, e.g., `PreCog_PoC_Notebooks` for general exploration and `PreCog_PoC_Source` for more structured code.
    3.  To create a notebook: Right-click within a folder (or the user folder) > **Create** > **Notebook**.
    4.  Name your notebook, choose **Python** as the default language, and select the cluster you created in Step 2 to attach it to. The notebook will execute its code on this selected cluster.
*   **Git Integration (Databricks Repos):**
    1.  In the sidebar, navigate to **Repos**.
    2.  Click **Add Repo**.
    3.  Provide the URL of your Git repository (e.g., from GitHub, GitLab, Azure Repos).
    4.  This will clone your project (including `src/`, `notebooks/`, `README.md`, etc.) into Databricks. You can then open, edit, and run notebooks directly from the Repo.
    5.  You can commit, push, pull, and manage branches directly within Databricks Repos, keeping your Databricks development in sync with your Git repository.
*   **How it runs/connects:** Notebooks execute code on their attached cluster. They will read data from your configured storage (Step 3) and write processed data to Delta tables (Step 5).

## Phase 3: Building the Data Pipeline & BI

### Step 5: Create Delta Tables

*   **What they are:** Tables stored in the Delta Lake format. Delta Lake is an open-source storage layer that brings ACID transactions, scalable metadata handling, and unifies streaming and batch data processing to data lakes built on top of your existing cloud storage.
*   **How to create (typically from a Databricks Notebook using PySpark):**
    ```python
    # Example: In a Databricks Python notebook
    # Assume 'spark' is the SparkSession (available by default in Databricks notebooks)
    # Assume 'input_df' is a Spark DataFrame containing your data to be saved
    
    # Define the path for your Delta table (preferably on mounted cloud storage)
    delta_table_path = "/mnt/precog_data/delta/social_media_feed"
    
    # Write the DataFrame to a Delta table
    input_df.write.format("delta").mode("overwrite").save(delta_table_path)
    
    # Optionally, create a SQL table reference in the metastore for easier querying
    # This makes the table discoverable via SQL queries
    spark.sql(f"CREATE TABLE IF NOT EXISTS precog_social_feed USING DELTA LOCATION '{delta_table_path}'")
    
    # To append data (e.g., in a streaming or incremental batch scenario):
    # new_data_df.write.format("delta").mode("append").save(delta_table_path)
    ```
*   **How it runs/connects:**
    *   Your data processing notebooks (running on the cluster) will read raw data, perform transformations, and write the results into these Delta tables.
    *   Delta tables become the reliable, versioned, and queryable source for your Business Intelligence (BI) dashboards, further analysis, and machine learning model training.

### Step 6: Set Up Databricks SQL Endpoint (for BI Dashboard)

*   **What it is:** A compute resource specifically optimized for running SQL queries. It powers your BI tools and dashboards, like the `Community Pulse Dashboard` for PreCog.
*   **How to create:**
    1.  Switch to the **SQL** persona in the Databricks Workspace (use the persona switcher in the bottom-left of the sidebar).
    2.  In the SQL persona sidebar, go to **SQL Endpoints** (this might also be named **Warehouses** depending on your Databricks version/cloud).
    3.  Click **Create SQL Endpoint**.
    4.  Configure it:
        *   **Name:** e.g., `precog-bi-sql-endpoint`
        *   **Cluster Size:** Choose a T-shirt size (e.g., Small, Medium, Large) based on expected query load. Start small for the PoC.
        *   **Auto Stop:** Enable this to automatically stop the endpoint after a period of inactivity to save costs.
        *   **Scaling:** Configure scaling options if needed.
*   **How it runs/connects:**
    *   This SQL endpoint runs SQL queries efficiently against your Delta tables (created in Step 5).
    *   You'll select this endpoint when building queries and dashboards in Databricks SQL.

### Step 7: Create a Dashboard in Databricks SQL

*   **What it is:** A collection of visualizations (charts, tables, key performance indicators) based on SQL queries against your Delta tables, providing insights into your data.
*   **How to create:**
    1.  Ensure you are in the **SQL** persona.
    2.  In the sidebar, go to **Dashboards**.
    3.  Click **Create Dashboard**.
    4.  Give your dashboard a name (e.g., `PreCog Community Pulse`).
    5.  Add widgets (visualizations) to your dashboard:
        *   Click **Add Visualization** or **Add Textbox**.
        *   For visualizations, you'll first write a SQL query. Select your SQL Endpoint (created in Step 6) to run the query against.
        *   Example Query: `SELECT location_approximation, AVG(sentiment_score) AS avg_sentiment, COUNT(*) AS num_reports FROM precog_social_feed WHERE report_date >= current_date() - INTERVAL 7 DAYS GROUP BY location_approximation`
        *   After running the query, choose a visualization type (bar chart, line chart, map, table, etc.) and configure its settings.
*   **How it runs/connects:** The dashboard executes its underlying SQL queries using the selected SQL Endpoint. The data is fetched from your Delta tables, and the visualizations are rendered in the dashboard. Dashboards can be set to refresh automatically.

## Phase 4: Automation (Optional but Recommended for Real-time Simulation & Production)

### Step 8: Schedule Notebooks as Jobs (for pipeline automation)

*   **What it is:** A way to run your Databricks notebooks (or other tasks like Python scripts, JARs) automatically on a schedule or triggered by events (e.g., new file arrival).
*   **How to create:**
    1.  Switch back to the **Data Science & Engineering** (or **Machine Learning**) persona.
    2.  In the sidebar, go to **Workflows** (this was previously called **Jobs**).
    3.  Click **Create Job**.
    4.  Give your job a name (e.g., `precog-hourly-data-ingestion`).
    5.  Configure a **Task**:
        *   **Task name:** e.g., `IngestAndProcessSocialFeed`
        *   **Type:** Select **Notebook**.
        *   **Source:** Choose the notebook that performs your data ingestion and processing logic.
        *   **Cluster:** Select the cluster (created in Step 2) on which this job should run. You can use an existing all-purpose cluster or configure a job-specific cluster (which is created on-demand for the job run and terminates afterward, often more cost-effective for scheduled tasks).
        *   **Parameters (Optional):** Pass parameters to your notebook if it's designed to accept them.
    6.  Set a **Schedule** (e.g., hourly, daily, or cron syntax for more complex schedules). For your PoC's real-time simulation, you might schedule a job to run frequently (e.g., every 5-15 minutes) if it's designed to process new files dropped by your `scripts/run_poc_simulation.py`.
    7.  Configure alerts for job success/failure if needed.
*   **How it runs/connects:** The Databricks job scheduler automatically triggers your specified notebook to run on the chosen cluster according to the defined schedule. This automates your data pipeline, ensuring your Delta tables (and consequently, your dashboards) are kept up-to-date with the latest data.

## Summary of Connections & Workflow

1.  **Databricks Workspace:** The overarching environment.
2.  **Data Sources** (Cloud Storage/DBFS): Hold raw input data.
3.  **Cluster (Compute):** Provides the processing power.
4.  **Databricks Repos:** Manages your version-controlled project code (notebooks, Python files).
5.  **Notebooks** (running on a Cluster, accessed via Workspace/Repos):
    *   Read data from **Data Sources**.
    *   Perform ETL (Extract, Transform, Load) and AI/ML processing.
    *   Write processed data into **Delta Tables**.
6.  **Delta Tables** (stored in Cloud Storage/DBFS): Store reliable, versioned, and queryable data.
7.  **Jobs (Workflows):** Automate the execution of **Notebooks** on **Clusters** to regularly update **Delta Tables**.
8.  **SQL Endpoint:** Provides dedicated compute for SQL queries against **Delta Tables**.
9.  **Databricks SQL Dashboards:** Query **Delta Tables** via the **SQL Endpoint** to visualize insights for users.

This structured approach will help you build a scalable and maintainable data and AI platform for the PreCog project on Databricks. For the hackathon, focus on getting a simplified version of this pipeline working to demonstrate your core PoC goals.