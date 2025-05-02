# NLP-Based To-Do List Extractor

This project is part of the Artificial Intelligence course. The goal is to develop a machine learning model that extracts structured information from natural language inputs for a to-do list application. 

## Problem Statement
Given a natural language task input by the user, the model extracts:
- Task description
- Date
- Time
- Priority

Example input:
> "Submit assignment by Monday 5pm, urgent"

Extracted output:
```json
{
  "task": "Submit assignment",
  "date": "Monday",
  "time": "5pm",
  "priority": "urgent"
}
