
# Flow

## Dev

### Setup
```bash
claude --dangerously-skip-permissions
````

- fill project information in CLAUDE.md
- Ask him to prepare a development plan
- Ask him to prepare basic folder structure
- Ask him to choose minimal dependencies
- Update CLAUDE.md


* **Python**

  ```bash
  claude "Generate a project structure for my 'Project Overview' using CLAUDE.md information with the following requirements:
  - Testing setup to check testing setup
  - add missing dependencies from 'Dependencies' section
  - Basic folder structure following best practices
  - Configuration files for testing
  - A simple README.md
  "
  ```

### Check installation

* **Python**

  ```bash
  poetry install
  poetry run python -m pytest
  ```

### Update `CLAUDE.md`

```bash
claude "Update @CLAUDE.md with current project folder structure"
```

```bash
claude "Update @CLAUDE.md with current project folder structure and new commands"
```

### Implementation flow

```bash
claude "Develop a part of this user story: [story] with corresponding unit tests: [part]"
```

```bash
claude "Develop a next part of current user story with corresponding minimal unit tests: [part]"
```

```bash
claude "Create a implementation plan for this 'Project Overview' locatecd in @CLAUDE.md.
- I want to have modular plan that will allow to work in small iterative steps.
- This is Proof of Concept application not production ready app.
- Include minimal necessary dependencies needed to develop the project
```

```bash
# Turn plan mode on

claude "Develop following part of the application from Phase [x]: [part].
Create minimal tests and at the end of your work summarize what you did and why this is important from project perspective.
```

```bash
claude "Refactor this code for better [performance/readability]: [code]"
```

### Workbook


```bash
claude "Check if you have the ability to run and test this application. Use @CLAUDE.md for project information."
```

(optional if plan, deps already exist)
```bash
claude "Based on information in @CLAUDE.md setup this project.
- Use already present environment setup, just extend it
- Install dependencies
- Prepare folder structure
At the end check if you have the ability to run and test this application.
```

```bash
claude "Create a implementation plan for this 'Project Overview' located in @CLAUDE.md.
- I want to have modular plan that will allow to work in small iterative steps.
- This is Proof of Concept application not production ready app.
- Include necessary dependencies needed to develop the project, if you do not have to implement something, use a library
  Choose lightweight and well known libraries
```

```bash
claude "Update @CLAUDE.md with current project folder structure, any new commands and dependencies"
```

```bash
claude "Develop following part of the application from: []
Create minimal tests and at the end of your work quickly summarize what you did and why and
what was not accomplished if that is the case."
```

```bash
claude "Check if your changes are aligned with overall Project goals and are moving the development in completion direction."
```