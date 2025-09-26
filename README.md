---
title: AutoHDL
emoji: 🚀
colorFrom: blue
colorTo: blue
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: AI agent which generates Verilog code
license: mit
---

# AutoHDL

AI agent which generates Verilog code

This project is a work-in-progress!

## Introduction

This is an attempt to build an AI agent which tries to generate syntactically correct
Verilog code from input text specifications.

When I'm done, the LLM will be able to call Verilog linter and simulation tools.
It will then use the tools output to self-correct the Verilog code it suggests.

For testing and illustration purposes, I'm using the MG-Verilog dataset from Georgia Tech.

## Tools

- `transformers` for LLM calls
- `outlines` for structured LLM output
- `streamlit` for pipeline visualization
