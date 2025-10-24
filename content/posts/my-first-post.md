---
date: '2025-10-24T13:58:55+02:00'
draft: false
title: 'Chain-of-Thought Prompting'
description: 'Paper-reading notes: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models'
ShowWordCount: true
ShowReadingTime: false
---

<aside>

**Chain of thought:**

- A series of intermediate **natural language reasoning steps** that lead to the final output — significantly improves the ability of large language models to perform complex reasoning.

**Chain-of-thought prompting:**

- A simple and broadly applicable method for
    - enhancing reasoning in language models.
    - Improving performance on a range of **arithmetic**, **commonsense**, and **symbolic**
    reasoning tasks.
</aside>

---

## Two simple methods to unlock the reasoning ability of LLMs

- **Thinking in steps helps:**
    
    When the model explains each step before the answer, it understands the problem better.
    
- **Learning from examples:**
    
    When we show a few examples with step-by-step answers, the model learns to do the same.
    

<aside>

**Few-shot prompting** means giving the model **a few examples** in the prompt to show it **how to do a task** before asking it to solve a new one.

</aside>

## What this paper do

- Combine these two ideas
    - help the language models **think step by step** to generate a clear and logical chain of ideas that **shows how they reach the final answer.**
- Given a prompt that consists of triples: **<input, chain of thought, output>**

## Why this method is important

- It doesn’t need a big training dataset.
- One model can do many different tasks without extra training.

<aside>

**Greedy decoding:** let the model **choose the most likely next word each time**

</aside>

## Result

1. It **only works well for giant models**, not smaller ones.
2. It **only works well for more-complicated problems**, not the simple ones.
3. Chain-of-thought prompting with big models gives results **as good as or better than** older methods that needed finetune **for each task**.

<aside>

**Ablation study:** It’s like testing which parts of your model really matter.

</aside>

## Related work

- Some methods make the **input part** of the prompt better — for example, adding clearer **instructions** before the question.
- But this paper does something **different (orthogonal)**: it improves the **output part**, by making the model **generate reasoning steps (chain of thought)** before giving the final answer.