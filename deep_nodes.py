"""
Deep analysis nodes for generating comprehensive, in-depth documentation.

These nodes extend the standard tutorial pipeline with:
- DeepAnalysisNode: Deep dive into each abstraction
- DesignPatternNode: Design pattern analysis
- ArchitectureOverviewNode: Global architecture overview with Mermaid diagrams
- CodeWalkthroughNode: Line-by-line code walkthrough
- TutorialSynthesisNode: Final synthesis into a complete document
"""

import os
import re
import yaml
from pocketflow import Node, BatchNode
from utils.call_llm import call_llm
from utils.semantic_chunks import (
    build_chunk_inventory,
    format_chunks_for_prompt,
    select_chunks_by_ids,
)

LLM_PLANNER_MAX_CHUNKS = 250
DEFAULT_TUTORIAL_LANGUAGE = "Chinese"


def _positive_int(value, default):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _load_yaml_block(response):
    text = response.strip()
    if "```yaml" in text:
        text = text.split("```yaml", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    data = yaml.safe_load(text)
    if data is None:
        raise ValueError("LLM returned empty YAML")
    return data


def get_content_for_indices(files_data, indices):
    content_map = {}
    for i in indices:
        if 0 <= i < len(files_data):
            path, content = files_data[i]
            content_map[f"{i} # {path}"] = content
    return content_map


# ============================================================================
# DeepAnalysisNode: Deep dive into each abstraction
# ============================================================================

class DeepAnalysisNode(BatchNode):
    """Performs deep analysis on each identified abstraction.

    For each abstraction, generates:
    - Design motivation and historical context
    - Trade-off analysis
    - Comparison with alternative approaches
    - Potential improvements
    - Real-world usage scenarios
    - Common pitfalls and best practices
    """

    def prep(self, shared):
        abstractions = shared["abstractions"]
        files_data = shared["files"]
        project_name = shared["project_name"]
        language = shared.get("language", DEFAULT_TUTORIAL_LANGUAGE)
        use_cache = shared.get("use_cache", True)

        items = []
        for i, abstraction in enumerate(abstractions):
            file_indices = abstraction.get("files", [])
            files_content = get_content_for_indices(files_data, file_indices)

            items.append({
                "index": i,
                "abstraction": abstraction,
                "files_content": files_content,
                "project_name": project_name,
                "language": language,
                "use_cache": use_cache,
            })

        print(f"Preparing deep analysis for {len(items)} abstractions...")
        return items

    def exec(self, item):
        abstraction = item["abstraction"]
        name = abstraction["name"]
        description = abstraction["description"]
        project_name = item["project_name"]
        language = item["language"]
        use_cache = item["use_cache"]

        print(f"  Deep analyzing: {name}")

        file_context = "\n\n".join(
            f"--- File: {path.split('# ')[1] if '# ' in path else path} ---\n{content}"
            for path, content in item["files_content"].items()
        )

        lang_instruction = ""
        if language.lower() != "english":
            lang_cap = language.capitalize()
            lang_instruction = f"IMPORTANT: Generate ALL content in **{lang_cap}**. Only keep code syntax in English.\n\n"

        prompt = f"""
{lang_instruction}For the project `{project_name}`, perform a deep analysis of the abstraction: "{name}"

Description:
{description}

Relevant Code:
{file_context if file_context else "No specific code provided."}

Provide a comprehensive analysis covering ALL of the following aspects:

1. **设计动机 (Design Motivation)**
   - 这个抽象解决什么具体问题？
   - 没有这个抽象会怎样？
   - 这个设计的灵感来源是什么？

2. **设计权衡 (Design Trade-offs)**
   - 这个设计选择了什么？放弃了什么？
   - 为什么做出这样的选择？
   - 这些权衡在什么场景下是值得的？

3. **替代方案对比 (Alternative Approaches)**
   - 列举 2-3 种替代实现方式
   - 每种替代方案的优缺点
   - 为什么当前方案被选中？

4. **设计模式分析 (Design Pattern Analysis)**
   - 使用了哪些设计模式？
   - 为什么选择这些模式？
   - 这些模式如何协作？

5. **潜在改进空间 (Potential Improvements)**
   - 当前设计的局限性
   - 可以优化的方向
   - 未来演进的可能性

6. **实际使用场景 (Real-world Usage Scenarios)**
   - 典型的使用场景
   - 边界情况和特殊用法
   - 反模式和需要避免的做法

7. **常见错误和陷阱 (Common Pitfalls)**
   - 初学者容易犯的错误
   - 如何避免这些错误
   - 调试技巧

8. **最佳实践 (Best Practices)**
   - 使用这个抽象的推荐方式
   - 性能优化建议
   - 与其他组件的配合方式

Format the output as YAML:

```yaml
design_motivation:
  problem: |
    详细描述解决的问题
  without_this: |
    没有这个抽象的后果
  inspiration: |
    设计灵感来源

trade_offs:
  chosen: |
    选择了什么
  sacrificed: |
    放弃了什么
  rationale: |
    选择理由
  scenarios: |
    适用场景

alternatives:
  - name: "替代方案1"
    pros: |
      优点
    cons: |
      缺点
  - name: "替代方案2"
    pros: |
      优点
    cons: |
      缺点

design_patterns:
  - pattern: "模式名称"
    usage: |
      如何使用
    reason: |
      选择原因

improvements:
  limitations: |
    当前局限性
  optimization: |
    优化方向
  evolution: |
    未来演进

usage_scenarios:
  typical: |
    典型场景
  edge_cases: |
    边界情况
  anti_patterns: |
    反模式

pitfalls:
  - mistake: "常见错误"
    solution: "解决方案"
  - mistake: "常见错误"
    solution: "解决方案"

best_practices:
  - practice: "最佳实践"
    reason: "原因"
  - practice: "最佳实践"
    reason: "原因"
```
"""
        response = call_llm(
            prompt,
            use_cache=(use_cache and self.cur_retry == 0),
            stage="deep.analysis",
            metadata={
                "project_name": project_name,
                "abstraction_name": name,
                "abstraction_index": item["index"],
            },
        )

        data = _load_yaml_block(response)
        return {
            "abstraction_index": item["index"],
            "abstraction_name": name,
            "analysis": data,
        }

    def post(self, shared, prep_res, exec_res_list):
        shared["deep_analyses"] = exec_res_list
        print(f"Completed deep analysis for {len(exec_res_list)} abstractions.")


# ============================================================================
# DesignPatternNode: Analyze design patterns used in the codebase
# ============================================================================

class DesignPatternNode(Node):
    """Identifies and analyzes design patterns used in the codebase.

    Generates:
    - List of design patterns found
    - Why each pattern was chosen
    - What problem each pattern solves
    - Trade-offs of using each pattern
    - Alternative patterns that could be used
    """

    def prep(self, shared):
        abstractions = shared["abstractions"]
        files_data = shared["files"]
        project_name = shared["project_name"]
        language = shared.get("language", DEFAULT_TUTORIAL_LANGUAGE)
        use_cache = shared.get("use_cache", True)

        # Collect all relevant file indices
        all_indices = set()
        for abstr in abstractions:
            all_indices.update(abstr.get("files", []))

        files_content = get_content_for_indices(files_data, sorted(all_indices))

        abstraction_info = []
        for i, abstr in enumerate(abstractions):
            abstraction_info.append(f"- {i}: {abstr['name']}: {abstr['description'][:100]}...")

        return {
            "abstraction_info": "\n".join(abstraction_info),
            "files_content": files_content,
            "project_name": project_name,
            "language": language,
            "use_cache": use_cache,
            "num_abstractions": len(abstractions),
        }

    def exec(self, prep_res):
        project_name = prep_res["project_name"]
        language = prep_res["language"]
        use_cache = prep_res["use_cache"]

        print("Analyzing design patterns...")

        file_context = "\n\n".join(
            f"--- File: {path.split('# ')[1] if '# ' in path else path} ---\n{content}"
            for path, content in prep_res["files_content"].items()
        )

        lang_instruction = ""
        if language.lower() != "english":
            lang_cap = language.capitalize()
            lang_instruction = f"IMPORTANT: Generate ALL content in **{lang_cap}**. Only keep code syntax and pattern names in English.\n\n"

        prompt = f"""
{lang_instruction}For the project `{project_name}`, analyze the design patterns used in the codebase.

Abstractions identified:
{prep_res["abstraction_info"]}

Relevant Code:
{file_context}

Identify ALL design patterns used in this codebase. For each pattern, provide:

1. **模式名称 (Pattern Name)** - Use the standard English name
2. **模式类型 (Pattern Type)** - Creational / Structural / Behavioral
3. **使用位置 (Where Used)** - Which files/modules use this pattern
4. **解决的问题 (Problem Solved)** - What problem does this pattern address
5. **实现方式 (Implementation)** - How is it implemented in this codebase
6. **选择理由 (Why Chosen)** - Why was this pattern selected
7. **代价 (Cost/Trade-offs)** - What are the downsides
8. **替代方案 (Alternatives)** - What other patterns could be used instead
9. **协作模式 (Pattern Collaboration)** - How does this pattern work with other patterns

Also identify:
- **架构模式 (Architectural Patterns)** - Overall architectural style (e.g., MVC, Pipeline, etc.)
- **惯用法 (Idioms)** - Language-specific idioms used

Format the output as YAML:

```yaml
architectural_patterns:
  - name: "Pipeline"
    description: |
      整体架构描述
    evidence: |
      代码证据

design_patterns:
  - name: "Observer"
    type: "Behavioral"
    locations:
      - "文件路径"
    problem: |
      解决的问题
    implementation: |
      实现方式
    rationale: |
      选择理由
    trade_offs: |
      代价
    alternatives:
      - "替代方案1"
      - "替代方案2"
    collaboration: |
      与其他模式的协作

idioms:
  - name: "惯用法名称"
    description: |
      描述
    example: |
      示例
```
"""
        response = call_llm(
            prompt,
            use_cache=(use_cache and self.cur_retry == 0),
            stage="design.patterns",
            metadata={
                "project_name": project_name,
                "num_abstractions": prep_res["num_abstractions"],
            },
        )

        data = _load_yaml_block(response)
        return data

    def post(self, shared, prep_res, exec_res):
        shared["design_patterns"] = exec_res
        print("Design pattern analysis complete.")


# ============================================================================
# ArchitectureOverviewNode: Generate global architecture overview
# ============================================================================

class ArchitectureOverviewNode(Node):
    """Generates a comprehensive architecture overview with detailed Mermaid diagrams.

    Produces:
    - System architecture diagram
    - Data flow diagram
    - Component dependency diagram
    - Key path analysis
    - Layer/Module organization
    """

    def prep(self, shared):
        abstractions = shared["abstractions"]
        relationships = shared["relationships"]
        files_data = shared["files"]
        project_name = shared["project_name"]
        language = shared.get("language", DEFAULT_TUTORIAL_LANGUAGE)
        use_cache = shared.get("use_cache", True)
        deep_analyses = shared.get("deep_analyses", [])
        design_patterns = shared.get("design_patterns", {})

        # Collect all relevant file indices
        all_indices = set()
        for abstr in abstractions:
            all_indices.update(abstr.get("files", []))

        files_content = get_content_for_indices(files_data, sorted(all_indices))

        abstraction_info = []
        for i, abstr in enumerate(abstractions):
            abstraction_info.append(f"- {i}: {abstr['name']}: {abstr['description']}")

        relationship_info = []
        for rel in relationships.get("details", []):
            from_name = abstractions[rel["from"]]["name"]
            to_name = abstractions[rel["to"]]["name"]
            relationship_info.append(f"- {from_name} --[{rel['label']}]--> {to_name}")

        # Include deep analysis summaries if available
        analysis_summary = ""
        if deep_analyses:
            analysis_summary = "\n\nDeep Analysis Summaries:\n"
            for da in deep_analyses:
                analysis = da.get("analysis", {})
                motivation = analysis.get("design_motivation", {})
                analysis_summary += f"\n- {da['abstraction_name']}: {motivation.get('problem', 'N/A')[:200]}"

        return {
            "abstraction_info": "\n".join(abstraction_info),
            "relationship_info": "\n".join(relationship_info),
            "files_content": files_content,
            "project_name": project_name,
            "language": language,
            "use_cache": use_cache,
            "summary": relationships.get("summary", ""),
            "analysis_summary": analysis_summary,
            "design_patterns": design_patterns,
        }

    def exec(self, prep_res):
        project_name = prep_res["project_name"]
        language = prep_res["language"]
        use_cache = prep_res["use_cache"]

        print("Generating architecture overview...")

        file_context = "\n\n".join(
            f"--- File: {path.split('# ')[1] if '# ' in path else path} ---\n{content[:3000]}"
            for path, content in list(prep_res["files_content"].items())[:20]
        )

        lang_instruction = ""
        if language.lower() != "english":
            lang_cap = language.capitalize()
            lang_instruction = f"IMPORTANT: Generate ALL text content in **{lang_cap}**. Keep Mermaid labels concise but in {lang_cap}.\n\n"

        prompt = f"""
{lang_instruction}For the project `{project_name}`, generate a comprehensive architecture overview.

Project Summary:
{prep_res["summary"]}

Abstractions:
{prep_res["abstraction_info"]}

Relationships:
{prep_res["relationship_info"]}
{prep_res["analysis_summary"]}

Relevant Code (truncated for context):
{file_context}

Generate a detailed architecture overview with MULTIPLE Mermaid diagrams. Include:

1. **系统架构概览 (System Architecture Overview)**
   - High-level architecture diagram showing all major components
   - Layer organization (if applicable)
   - External dependencies

2. **数据流图 (Data Flow Diagram)**
   - How data moves through the system
   - Input/output at each stage
   - Data transformations

3. **组件依赖关系图 (Component Dependency Diagram)**
   - Dependencies between components
   - Circular dependencies (if any)
   - Dependency direction

4. **关键路径分析 (Critical Path Analysis)**
   - Main execution paths
   - Error handling paths
   - Performance-critical paths

5. **模块组织 (Module Organization)**
   - Directory structure explanation
   - Module responsibilities
   - Interface boundaries

IMPORTANT: Generate MULTIPLE Mermaid diagrams (at least 3-4), each focused on a different aspect.
Use ```mermaid``` blocks. Keep diagrams readable (max 15 nodes per diagram).

Format the output as YAML:

```yaml
overview: |
  系统整体架构描述...

architecture_diagram: |
  ```mermaid
  flowchart TD
    ...
  ```

data_flow_diagram: |
  ```mermaid
  flowchart LR
    ...
  ```

dependency_diagram: |
  ```mermaid
  graph TD
    ...
  ```

critical_paths:
  - name: "主执行路径"
    description: |
      描述
    diagram: |
      ```mermaid
      sequenceDiagram
        ...
      ```

module_organization: |
  模块组织描述...

layers:
  - name: "Layer Name"
    description: |
      描述
    components:
      - "Component1"
      - "Component2"
```
"""
        response = call_llm(
            prompt,
            use_cache=(use_cache and self.cur_retry == 0),
            stage="architecture.overview",
            metadata={
                "project_name": project_name,
            },
        )

        data = _load_yaml_block(response)
        return data

    def post(self, shared, prep_res, exec_res):
        shared["architecture_overview"] = exec_res
        print("Architecture overview generated.")


# ============================================================================
# CodeWalkthroughNode: Line-by-line code walkthrough
# ============================================================================

class CodeWalkthroughNode(BatchNode):
    """Generates detailed code walkthroughs for key files.

    For each key file, generates:
    - Execution flow explanation
    - Key variable meanings
    - Important design decisions
    - Edge case handling
    - Line-by-line commentary
    """

    def prep(self, shared):
        abstractions = shared["abstractions"]
        files_data = shared["files"]
        project_name = shared["project_name"]
        language = shared.get("language", DEFAULT_TUTORIAL_LANGUAGE)
        use_cache = shared.get("use_cache", True)

        # Identify key files (files referenced by multiple abstractions)
        file_ref_count = {}
        for abstr in abstractions:
            for idx in abstr.get("files", []):
                file_ref_count[idx] = file_ref_count.get(idx, 0) + 1

        # Select top key files (referenced by at least 1 abstraction, max 10)
        key_files = sorted(file_ref_count.items(), key=lambda x: -x[1])[:10]

        items = []
        for file_idx, ref_count in key_files:
            if 0 <= file_idx < len(files_data):
                path, content = files_data[file_idx]
                # Find which abstractions reference this file
                related_abstractions = []
                for abstr in abstractions:
                    if file_idx in abstr.get("files", []):
                        related_abstractions.append(abstr["name"])

                items.append({
                    "file_index": file_idx,
                    "file_path": path,
                    "file_content": content,
                    "ref_count": ref_count,
                    "related_abstractions": related_abstractions,
                    "project_name": project_name,
                    "language": language,
                    "use_cache": use_cache,
                })

        print(f"Preparing code walkthrough for {len(items)} key files...")
        return items

    def exec(self, item):
        file_path = item["file_path"]
        file_content = item["file_content"]
        project_name = item["project_name"]
        language = item["language"]
        use_cache = item["use_cache"]
        related_abstractions = item["related_abstractions"]

        print(f"  Walking through: {file_path}")

        # Truncate very large files
        if len(file_content) > 15000:
            file_content = file_content[:15000] + "\n... (truncated)"

        lang_instruction = ""
        if language.lower() != "english":
            lang_cap = language.capitalize()
            lang_instruction = f"IMPORTANT: Generate ALL explanatory text in **{lang_cap}**. Keep code as-is.\n\n"

        prompt = f"""
{lang_instruction}For the project `{project_name}`, provide a detailed code walkthrough for:

File: {file_path}
Related abstractions: {', '.join(related_abstractions)}

Code:
```python
{file_content}
```

Provide a comprehensive walkthrough covering:

1. **文件概述 (File Overview)**
   - 文件的主要职责
   - 在整体架构中的位置
   - 与其他文件的关系

2. **执行流程 (Execution Flow)**
   - 主要执行路径
   - 使用 Mermaid 流程图展示

3. **关键变量和数据结构 (Key Variables & Data Structures)**
   - 重要的变量及其含义
   - 数据结构的设计理由
   - 状态管理方式

4. **重要设计决策 (Important Design Decisions)**
   - 关键的设计选择
   - 为什么这样设计
   - 权衡考虑

5. **边界情况处理 (Edge Case Handling)**
   - 错误处理机制
   - 边界条件检查
   - 防御性编程措施

6. **逐段详解 (Section-by-Section Walkthrough)**
   - 对代码的每个重要部分进行解释
   - 包含代码片段和解释

Format the output as YAML:

```yaml
file_overview: |
  文件概述...

execution_flow: |
  执行流程描述...

execution_flow_diagram: |
  ```mermaid
  flowchart TD
    ...
  ```

key_variables:
  - name: "variable_name"
    meaning: |
      变量含义
    design_reason: |
      设计理由

design_decisions:
  - decision: |
      设计决策
    rationale: |
      决策理由
    trade_offs: |
      权衡考虑

edge_cases:
  - case: "边界情况"
    handling: |
      处理方式

section_walkthrough:
  - section: "代码段描述"
    code_snippet: |
      关键代码
    explanation: |
      详细解释
```
"""
        response = call_llm(
            prompt,
            use_cache=(use_cache and self.cur_retry == 0),
            stage="code.walkthrough",
            metadata={
                "project_name": project_name,
                "file_path": file_path,
                "file_index": item["file_index"],
            },
        )

        data = _load_yaml_block(response)
        return {
            "file_path": file_path,
            "file_index": item["file_index"],
            "walkthrough": data,
        }

    def post(self, shared, prep_res, exec_res_list):
        shared["code_walkthroughs"] = exec_res_list
        print(f"Completed code walkthrough for {len(exec_res_list)} files.")


# ============================================================================
# EnhancedWriteChaptersNode: Write more detailed chapters
# ============================================================================

class EnhancedWriteChaptersNode(BatchNode):
    """Writes enhanced, much more detailed tutorial chapters.

    Each chapter includes:
    - Concept origin and background
    - Why this concept is needed
    - Multiple real-world examples
    - Common errors and pitfalls
    - Best practices
    - Connections to other concepts
    - Deep analysis integration
    - Code walkthrough integration
    """

    def prep(self, shared):
        chapter_order = shared["chapter_order"]
        abstractions = shared["abstractions"]
        files_data = shared["files"]
        project_name = shared["project_name"]
        language = shared.get("language", DEFAULT_TUTORIAL_LANGUAGE)
        use_cache = shared.get("use_cache", True)
        relationships = shared["relationships"]
        deep_analyses = shared.get("deep_analyses", [])
        design_patterns = shared.get("design_patterns", {})
        architecture_overview = shared.get("architecture_overview", {})
        code_walkthroughs = shared.get("code_walkthroughs", [])

        # Build chapter listing for cross-references
        all_chapters = []
        chapter_filenames = {}
        for i, abstraction_index in enumerate(chapter_order):
            if 0 <= abstraction_index < len(abstractions):
                chapter_num = i + 1
                chapter_name = abstractions[abstraction_index]["name"]
                safe_name = "".join(
                    c if c.isalnum() else "_" for c in chapter_name
                ).lower()
                filename = f"{i+1:02d}_{safe_name}.md"
                all_chapters.append(f"{chapter_num}. [{chapter_name}]({filename})")
                chapter_filenames[abstraction_index] = {
                    "num": chapter_num,
                    "name": chapter_name,
                    "filename": filename,
                }

        full_chapter_listing = "\n".join(all_chapters)

        # Build deep analysis lookup
        analysis_lookup = {}
        for da in deep_analyses:
            analysis_lookup[da["abstraction_index"]] = da["analysis"]

        # Build code walkthrough lookup
        walkthrough_lookup = {}
        for cw in code_walkthroughs:
            walkthrough_lookup[cw["file_index"]] = cw["walkthrough"]

        self.chapters_written_so_far = []

        items = []
        for i, abstraction_index in enumerate(chapter_order):
            if 0 <= abstraction_index < len(abstractions):
                abstraction = abstractions[abstraction_index]
                file_indices = abstraction.get("files", [])
                files_content = get_content_for_indices(files_data, file_indices)

                # Get related walkthroughs
                related_walkthroughs = []
                for idx in file_indices:
                    if idx in walkthrough_lookup:
                        related_walkthroughs.append({
                            "file_path": files_data[idx][0],
                            "walkthrough": walkthrough_lookup[idx],
                        })

                prev_chapter = chapter_filenames[chapter_order[i-1]] if i > 0 else None
                next_chapter = chapter_filenames[chapter_order[i+1]] if i < len(chapter_order) - 1 else None

                items.append({
                    "chapter_num": i + 1,
                    "abstraction_index": abstraction_index,
                    "abstraction": abstraction,
                    "files_content": files_content,
                    "project_name": project_name,
                    "language": language,
                    "use_cache": use_cache,
                    "full_chapter_listing": full_chapter_listing,
                    "chapter_filenames": chapter_filenames,
                    "prev_chapter": prev_chapter,
                    "next_chapter": next_chapter,
                    "summary": relationships.get("summary", ""),
                    "deep_analysis": analysis_lookup.get(abstraction_index, {}),
                    "related_walkthroughs": related_walkthroughs,
                    "design_patterns": design_patterns,
                })

        print(f"Preparing to write {len(items)} enhanced chapters...")
        return items

    def exec(self, item):
        abstraction = item["abstraction"]
        name = abstraction["name"]
        description = abstraction["description"]
        chapter_num = item["chapter_num"]
        project_name = item["project_name"]
        language = item["language"]
        use_cache = item["use_cache"]

        print(f"  Writing enhanced chapter {chapter_num}: {name}")

        file_context = "\n\n".join(
            f"--- File: {path.split('# ')[1] if '# ' in path else path} ---\n{content}"
            for path, content in item["files_content"].items()
        )

        # Format deep analysis
        deep_analysis_text = ""
        if item["deep_analysis"]:
            da = item["deep_analysis"]
            deep_analysis_text = f"""
Deep Analysis:
- Design Motivation: {da.get('design_motivation', {}).get('problem', 'N/A')}
- Trade-offs: {da.get('trade_offs', {}).get('rationale', 'N/A')}
- Common Pitfalls: {', '.join(p.get('mistake', '') for p in da.get('pitfalls', []))}
- Best Practices: {', '.join(p.get('practice', '') for p in da.get('best_practices', []))}
"""

        # Format walkthroughs
        walkthrough_text = ""
        if item["related_walkthroughs"]:
            walkthrough_text = "\n\nCode Walkthrough Summaries:\n"
            for wt in item["related_walkthroughs"][:3]:
                overview = wt["walkthrough"].get("file_overview", "")
                walkthrough_text += f"\n- {wt['file_path']}: {overview[:200]}"

        previous_summary = "\n---\n".join(self.chapters_written_so_far)

        lang_instruction = ""
        if language.lower() != "english":
            lang_cap = language.capitalize()
            lang_instruction = f"IMPORTANT: Write this ENTIRE chapter in **{lang_cap}**. Only keep code syntax in English.\n\n"

        prompt = f"""
{lang_instruction}Write a VERY detailed, in-depth tutorial chapter for the project `{project_name}` about: "{name}"

This is Chapter {chapter_num}.

Concept Details:
- Name: {name}
- Description: {description}

Project Summary:
{item["summary"]}

Complete Tutorial Structure:
{item["full_chapter_listing"]}

Context from previous chapters:
{previous_summary if previous_summary else "This is the first chapter."}

{deep_analysis_text}

{walkthrough_text}

Relevant Code:
{file_context}

Write an EXTREMELY detailed chapter that covers ALL of the following:

## Chapter Structure Requirements:

### 1. 概念起源与背景 (Origin & Background)
- 这个概念的历史渊源
- 为什么软件工程需要这个概念
- 这个概念解决了什么根本问题
- 用一个生活中的类比来解释

### 2. 核心概念详解 (Core Concept Deep Dive)
- 从最简单的例子开始
- 逐步增加复杂度
- 每一步都要有代码示例
- 每个代码示例都要有详细的解释

### 3. 设计决策分析 (Design Decision Analysis)
- 为什么这样设计而不是那样设计
- 设计时考虑了哪些权衡
- 这些权衡在什么场景下是值得的
- 与其他方案的对比

### 4. 实际案例分析 (Real-world Case Studies)
- 至少 3 个不同复杂度的案例
- 从简单到复杂排列
- 每个案例都要有完整的代码
- 解释每个案例的关键点

### 5. 内部实现揭秘 (Implementation Internals)
- 逐行解释关键代码
- 使用 Mermaid 序列图展示调用流程
- 解释重要的设计模式
- 数据如何在内部流转

### 6. 常见错误与陷阱 (Common Mistakes & Pitfalls)
- 至少 5 个常见错误
- 每个错误的错误代码示例
- 正确的写法
- 为什么会犯这个错误
- 如何避免

### 7. 最佳实践 (Best Practices)
- 至少 5 条最佳实践
- 每条实践的代码示例
- 为什么这是最佳实践
- 什么时候可以例外

### 8. 高级主题 (Advanced Topics)
- 进阶用法
- 性能优化技巧
- 与其他概念的深度集成
- 边界情况处理

### 9. 与其他概念的关联 (Connections to Other Concepts)
- 这个概念如何与其他概念协作
- 在整体架构中的位置
- 使用 Markdown 链接引用其他章节

### 10. 总结与回顾 (Summary & Recap)
- 关键要点总结
- 常见问题解答
- 下一步学习建议

## Format Requirements:
- Use Markdown with proper headings (##, ###, ####)
- Use ```mermaid``` blocks for diagrams (at least 2-3 per chapter)
- Use > blockquotes for important notes
- Use **bold** and *italic* for emphasis
- Each code block should be BELOW 10 lines
- Include input/output examples for code snippets
- Use tables for comparisons

Output ONLY the Markdown content for this chapter.
"""
        chapter_content = call_llm(
            prompt,
            use_cache=(use_cache and self.cur_retry == 0),
            stage="enhanced.write.chapter",
            metadata={
                "project_name": project_name,
                "chapter_num": chapter_num,
                "abstraction_name": name,
            },
        )

        # Ensure proper heading
        actual_heading = f"# Chapter {chapter_num}: {name}"
        if not chapter_content.strip().startswith(f"# Chapter {chapter_num}"):
            lines = chapter_content.strip().split("\n")
            if lines and lines[0].strip().startswith("#"):
                lines[0] = actual_heading
                chapter_content = "\n".join(lines)
            else:
                chapter_content = f"{actual_heading}\n\n{chapter_content}"

        self.chapters_written_so_far.append(chapter_content)
        return chapter_content

    def post(self, shared, prep_res, exec_res_list):
        shared["enhanced_chapters"] = exec_res_list
        del self.chapters_written_so_far
        print(f"Finished writing {len(exec_res_list)} enhanced chapters.")


# ============================================================================
# TutorialSynthesisNode: Final synthesis into a complete document
# ============================================================================

class TutorialSynthesisNode(Node):
    """Synthesizes all analysis results into a comprehensive final document.

    Generates:
    - Quick start guide
    - Deep understanding series
    - Advanced topics
    - Reference materials
    - Complete index with all sections
    """

    def prep(self, shared):
        project_name = shared["project_name"]
        language = shared.get("language", DEFAULT_TUTORIAL_LANGUAGE)
        use_cache = shared.get("use_cache", True)
        relationships = shared["relationships"]
        abstractions = shared["abstractions"]
        chapter_order = shared["chapter_order"]
        deep_analyses = shared.get("deep_analyses", [])
        design_patterns = shared.get("design_patterns", {})
        architecture_overview = shared.get("architecture_overview", {})
        code_walkthroughs = shared.get("code_walkthroughs", [])
        enhanced_chapters = shared.get("enhanced_chapters", [])
        repo_url = shared.get("repo_url", "")

        return {
            "project_name": project_name,
            "language": language,
            "use_cache": use_cache,
            "summary": relationships.get("summary", ""),
            "abstractions": abstractions,
            "chapter_order": chapter_order,
            "deep_analyses": deep_analyses,
            "design_patterns": design_patterns,
            "architecture_overview": architecture_overview,
            "code_walkthroughs": code_walkthroughs,
            "enhanced_chapters": enhanced_chapters,
            "repo_url": repo_url,
        }

    def exec(self, prep_res):
        project_name = prep_res["project_name"]
        language = prep_res["language"]
        use_cache = prep_res["use_cache"]

        print("Synthesizing final tutorial document...")

        # Build summaries for the synthesis prompt
        abstraction_summary = "\n".join(
            f"- {a['name']}: {a['description'][:100]}..."
            for a in prep_res["abstractions"]
        )

        deep_analysis_summary = ""
        if prep_res["deep_analyses"]:
            deep_analysis_summary = "\n\nDeep Analysis Available:\n"
            for da in prep_res["deep_analyses"]:
                deep_analysis_summary += f"- {da['abstraction_name']}\n"

        pattern_summary = ""
        if prep_res["design_patterns"]:
            patterns = prep_res["design_patterns"].get("design_patterns", [])
            if patterns:
                pattern_summary = "\n\nDesign Patterns Found:\n"
                for p in patterns:
                    pattern_summary += f"- {p.get('name', 'Unknown')}: {p.get('problem', '')[:100]}\n"

        lang_instruction = ""
        if language.lower() != "english":
            lang_cap = language.capitalize()
            lang_instruction = f"IMPORTANT: Generate ALL content in **{lang_cap}**. Only keep code syntax in English.\n\n"

        prompt = f"""
{lang_instruction}For the project `{project_name}`, create a comprehensive tutorial synthesis document.

Project Summary:
{prep_res["summary"]}

Abstractions:
{abstraction_summary}
{deep_analysis_summary}
{pattern_summary}

This synthesis document should serve as the ENTRY POINT for the entire tutorial. It should include:

1. **快速入门指南 (Quick Start Guide)**
   - 5-minute overview
   - Key concepts in one sentence each
   - "Start here" for different reader levels

2. **深入理解系列 (Deep Understanding Series)**
   - Reading order recommendation
   - Prerequisites for each section
   - Learning path visualization

3. **架构全景 (Architecture Panorama)**
   - High-level architecture summary
   - Key architectural decisions
   - System boundaries and interfaces

4. **设计模式速查 (Design Patterns Quick Reference)**
   - Pattern name → Where used → Why
   - Quick lookup table

5. **常见问题 (FAQ)**
   - At least 10 common questions
   - Concise answers with code examples

6. **进阶主题 (Advanced Topics)**
   - Performance optimization
   - Extensibility points
   - Contributing guidelines

7. **参考资料 (References)**
   - Glossary of terms
   - Further reading
   - Related projects

8. **学习路径图 (Learning Path Diagram)**
   - Mermaid diagram showing recommended reading order
   - Dependencies between chapters

Format the output as YAML:

```yaml
quick_start: |
  快速入门指南内容...

learning_paths:
  beginner:
    description: "初学者路径"
    chapters: [0, 1, 2]
  intermediate:
    description: "进阶路径"
    chapters: [3, 4, 5]
  advanced:
    description: "高级路径"
    chapters: [6, 7, 8]

architecture_summary: |
  架构全景描述...

pattern_quick_ref:
  - pattern: "Pattern Name"
    location: "Where used"
    reason: "Why chosen"

faq:
  - question: "问题"
    answer: "答案"

advanced_topics:
  - topic: "主题"
    description: "描述"

glossary:
  - term: "术语"
    definition: "定义"

learning_path_diagram: |
  ```mermaid
  flowchart TD
    ...
  ```
```
"""
        response = call_llm(
            prompt,
            use_cache=(use_cache and self.cur_retry == 0),
            stage="tutorial.synthesis",
            metadata={
                "project_name": project_name,
            },
        )

        data = _load_yaml_block(response)
        return data

    def post(self, shared, prep_res, exec_res):
        shared["tutorial_synthesis"] = exec_res
        print("Tutorial synthesis complete.")


# ============================================================================
# DeepCombineTutorial: Combine all enhanced content into final output
# ============================================================================

class DeepCombineTutorial(Node):
    """Combines all enhanced analysis into the final tutorial output.

    Generates:
    - Enhanced index with all sections
    - Architecture overview page
    - Design patterns page
    - Enhanced chapters
    - Code walkthrough pages
    - Synthesis/Quick start page
    """

    def prep(self, shared):
        project_name = shared["project_name"]
        output_base_dir = shared.get("output_dir", "output")
        output_path = os.path.join(output_base_dir, project_name)
        repo_url = shared.get("repo_url", "")
        language = shared.get("language", DEFAULT_TUTORIAL_LANGUAGE)

        relationships = shared["relationships"]
        abstractions = shared["abstractions"]
        chapter_order = shared["chapter_order"]
        enhanced_chapters = shared.get("enhanced_chapters", [])
        architecture_overview = shared.get("architecture_overview", {})
        design_patterns = shared.get("design_patterns", {})
        deep_analyses = shared.get("deep_analyses", [])
        code_walkthroughs = shared.get("code_walkthroughs", [])
        tutorial_synthesis = shared.get("tutorial_synthesis", {})

        # Generate Mermaid diagram
        mermaid_lines = ["flowchart TD"]
        for i, abstr in enumerate(abstractions):
            node_id = f"A{i}"
            sanitized_name = abstr["name"].replace('"', "")
            mermaid_lines.append(f'    {node_id}["{sanitized_name}"]')
        for rel in relationships.get("details", []):
            from_id = f"A{rel['from']}"
            to_id = f"A{rel['to']}"
            edge_label = rel["label"].replace('"', "").replace("\n", " ")[:30]
            mermaid_lines.append(f'    {from_id} -- "{edge_label}" --> {to_id}')

        mermaid_diagram = "\n".join(mermaid_lines)

        return {
            "output_path": output_path,
            "project_name": project_name,
            "repo_url": repo_url,
            "language": language,
            "summary": relationships.get("summary", ""),
            "abstractions": abstractions,
            "chapter_order": chapter_order,
            "enhanced_chapters": enhanced_chapters,
            "architecture_overview": architecture_overview,
            "design_patterns": design_patterns,
            "deep_analyses": deep_analyses,
            "code_walkthroughs": code_walkthroughs,
            "tutorial_synthesis": tutorial_synthesis,
            "mermaid_diagram": mermaid_diagram,
        }

    def exec(self, prep_res):
        output_path = prep_res["output_path"]
        project_name = prep_res["project_name"]

        print(f"Combining enhanced tutorial into: {output_path}")
        os.makedirs(output_path, exist_ok=True)

        # Build chapter files mapping
        chapter_files = []
        chapter_filenames = {}
        for i, abstraction_index in enumerate(prep_res["chapter_order"]):
            if 0 <= abstraction_index < len(prep_res["abstractions"]):
                name = prep_res["abstractions"][abstraction_index]["name"]
                safe_name = "".join(c if c.isalnum() else "_" for c in name).lower()
                filename = f"{i+1:02d}_{safe_name}.md"
                chapter_filenames[abstraction_index] = {
                    "num": i + 1,
                    "name": name,
                    "filename": filename,
                }

        # Write enhanced chapters
        for i, content in enumerate(prep_res["enhanced_chapters"]):
            abstraction_index = prep_res["chapter_order"][i]
            if abstraction_index in chapter_filenames:
                filename = chapter_filenames[abstraction_index]["filename"]
                filepath = os.path.join(output_path, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                chapter_files.append({"filename": filename, "name": chapter_filenames[abstraction_index]["name"]})
                print(f"  - Wrote {filename}")

        # Write architecture overview
        arch_content = self._generate_architecture_page(prep_res)
        arch_path = os.path.join(output_path, "00_architecture_overview.md")
        with open(arch_path, "w", encoding="utf-8") as f:
            f.write(arch_content)
        print(f"  - Wrote 00_architecture_overview.md")

        # Write design patterns page
        patterns_content = self._generate_patterns_page(prep_res)
        patterns_path = os.path.join(output_path, "00_design_patterns.md")
        with open(patterns_path, "w", encoding="utf-8") as f:
            f.write(patterns_content)
        print(f"  - Wrote 00_design_patterns.md")

        # Write deep analysis pages
        for da in prep_res["deep_analyses"]:
            analysis_content = self._generate_analysis_page(da, prep_res)
            idx = da["abstraction_index"]
            name = da["abstraction_name"]
            safe_name = "".join(c if c.isalnum() else "_" for c in name).lower()
            filename = f"deep_{safe_name}.md"
            filepath = os.path.join(output_path, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(analysis_content)
            print(f"  - Wrote {filename}")

        # Write code walkthrough pages
        for cw in prep_res["code_walkthroughs"]:
            walkthrough_content = self._generate_walkthrough_page(cw, prep_res)
            file_path = cw["file_path"]
            safe_name = "".join(c if c.isalnum() else "_" for c in file_path).lower()
            filename = f"walkthrough_{safe_name}.md"
            filepath = os.path.join(output_path, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(walkthrough_content)
            print(f"  - Wrote {filename}")

        # Write synthesis/quick start page
        synthesis_content = self._generate_synthesis_page(prep_res)
        synthesis_path = os.path.join(output_path, "00_quick_start.md")
        with open(synthesis_path, "w", encoding="utf-8") as f:
            f.write(synthesis_content)
        print(f"  - Wrote 00_quick_start.md")

        # Write enhanced index
        index_content = self._generate_index(prep_res, chapter_files)
        index_path = os.path.join(output_path, "index.md")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(index_content)
        print(f"  - Wrote index.md")

        return output_path

    def _generate_index(self, prep_res, chapter_files):
        project_name = prep_res["project_name"]
        repo_url = prep_res["repo_url"]

        content = f"# 深入理解 {project_name}\n\n"
        content += f"{prep_res['summary']}\n\n"
        content += f"**源码仓库:** [{repo_url}]({repo_url})\n\n"

        content += "## 快速开始\n\n"
        content += "- [快速入门指南](00_quick_start.md)\n\n"

        content += "## 架构与设计\n\n"
        content += "- [架构概览](00_architecture_overview.md)\n"
        content += "- [设计模式分析](00_design_patterns.md)\n\n"

        content += "## 章节目录\n\n"
        content += "```mermaid\n" + prep_res["mermaid_diagram"] + "\n```\n\n"

        for cf in chapter_files:
            content += f"- [{cf['name']}]({cf['filename']})\n"

        content += "\n## 深度分析\n\n"
        for da in prep_res["deep_analyses"]:
            name = da["abstraction_name"]
            safe_name = "".join(c if c.isalnum() else "_" for c in name).lower()
            content += f"- [{name} 深度分析](deep_{safe_name}.md)\n"

        content += "\n## 代码走读\n\n"
        for cw in prep_res["code_walkthroughs"]:
            file_path = cw["file_path"]
            safe_name = "".join(c if c.isalnum() else "_" for c in file_path).lower()
            content += f"- [{file_path}](walkthrough_{safe_name}.md)\n"

        content += "\n---\n\nGenerated by AI Codebase Knowledge Builder (Deep Analysis Mode)"
        return content

    def _generate_architecture_page(self, prep_res):
        arch = prep_res["architecture_overview"]
        project_name = prep_res["project_name"]

        content = f"# {project_name} 架构概览\n\n"
        content += arch.get("overview", "") + "\n\n"

        content += "## 系统架构图\n\n"
        content += arch.get("architecture_diagram", "") + "\n\n"

        content += "## 数据流图\n\n"
        content += arch.get("data_flow_diagram", "") + "\n\n"

        content += "## 组件依赖关系\n\n"
        content += arch.get("dependency_diagram", "") + "\n\n"

        if arch.get("layers"):
            content += "## 分层架构\n\n"
            for layer in arch["layers"]:
                content += f"### {layer.get('name', 'Layer')}\n\n"
                content += layer.get("description", "") + "\n\n"

        content += "## 模块组织\n\n"
        content += arch.get("module_organization", "") + "\n\n"

        if arch.get("critical_paths"):
            content += "## 关键路径\n\n"
            for path in arch["critical_paths"]:
                content += f"### {path.get('name', 'Path')}\n\n"
                content += path.get("description", "") + "\n\n"
                content += path.get("diagram", "") + "\n\n"

        content += "\n---\n\n[返回目录](index.md)"
        return content

    def _generate_patterns_page(self, prep_res):
        patterns = prep_res["design_patterns"]
        project_name = prep_res["project_name"]

        content = f"# {project_name} 设计模式分析\n\n"

        if patterns.get("architectural_patterns"):
            content += "## 架构模式\n\n"
            for ap in patterns["architectural_patterns"]:
                content += f"### {ap.get('name', 'Pattern')}\n\n"
                content += ap.get("description", "") + "\n\n"
                content += f"**代码证据:**\n{ap.get('evidence', '')}\n\n"

        if patterns.get("design_patterns"):
            content += "## 设计模式\n\n"
            for dp in patterns["design_patterns"]:
                content += f"### {dp.get('name', 'Pattern')}\n\n"
                content += f"**类型:** {dp.get('type', 'Unknown')}\n\n"
                content += f"**解决的问题:**\n{dp.get('problem', '')}\n\n"
                content += f"**实现方式:**\n{dp.get('implementation', '')}\n\n"
                content += f"**选择理由:**\n{dp.get('rationale', '')}\n\n"
                content += f"**代价:**\n{dp.get('trade_offs', '')}\n\n"
                if dp.get("alternatives"):
                    content += "**替代方案:**\n"
                    for alt in dp["alternatives"]:
                        content += f"- {alt}\n"
                    content += "\n"

        if patterns.get("idioms"):
            content += "## 惯用法\n\n"
            for idiom in patterns["idioms"]:
                content += f"### {idiom.get('name', 'Idiom')}\n\n"
                content += idiom.get("description", "") + "\n\n"

        content += "\n---\n\n[返回目录](index.md)"
        return content

    def _generate_analysis_page(self, da, prep_res):
        analysis = da.get("analysis", {})
        name = da["abstraction_name"]

        content = f"# {name} 深度分析\n\n"

        # Design Motivation
        motivation = analysis.get("design_motivation", {})
        content += "## 设计动机\n\n"
        content += f"### 解决的问题\n\n{motivation.get('problem', '')}\n\n"
        content += f"### 没有这个抽象会怎样\n\n{motivation.get('without_this', '')}\n\n"
        content += f"### 设计灵感\n\n{motivation.get('inspiration', '')}\n\n"

        # Trade-offs
        tradeoffs = analysis.get("trade_offs", {})
        content += "## 设计权衡\n\n"
        content += f"### 选择了什么\n\n{tradeoffs.get('chosen', '')}\n\n"
        content += f"### 放弃了什么\n\n{tradeoffs.get('sacrificed', '')}\n\n"
        content += f"### 选择理由\n\n{tradeoffs.get('rationale', '')}\n\n"

        # Alternatives
        if analysis.get("alternatives"):
            content += "## 替代方案对比\n\n"
            for alt in analysis["alternatives"]:
                content += f"### {alt.get('name', 'Alternative')}\n\n"
                content += f"**优点:**\n{alt.get('pros', '')}\n\n"
                content += f"**缺点:**\n{alt.get('cons', '')}\n\n"

        # Design Patterns
        if analysis.get("design_patterns"):
            content += "## 使用的设计模式\n\n"
            for dp in analysis["design_patterns"]:
                content += f"### {dp.get('pattern', 'Pattern')}\n\n"
                content += f"**使用方式:**\n{dp.get('usage', '')}\n\n"
                content += f"**选择原因:**\n{dp.get('reason', '')}\n\n"

        # Improvements
        improvements = analysis.get("improvements", {})
        content += "## 潜在改进空间\n\n"
        content += f"### 当前局限性\n\n{improvements.get('limitations', '')}\n\n"
        content += f"### 优化方向\n\n{improvements.get('optimization', '')}\n\n"
        content += f"### 未来演进\n\n{improvements.get('evolution', '')}\n\n"

        # Usage Scenarios
        usage = analysis.get("usage_scenarios", {})
        content += "## 实际使用场景\n\n"
        content += f"### 典型场景\n\n{usage.get('typical', '')}\n\n"
        content += f"### 边界情况\n\n{usage.get('edge_cases', '')}\n\n"
        content += f"### 反模式\n\n{usage.get('anti_patterns', '')}\n\n"

        # Pitfalls
        if analysis.get("pitfalls"):
            content += "## 常见错误和陷阱\n\n"
            for pitfall in analysis["pitfalls"]:
                content += f"### {pitfall.get('mistake', 'Error')}\n\n"
                content += f"**解决方案:**\n{pitfall.get('solution', '')}\n\n"

        # Best Practices
        if analysis.get("best_practices"):
            content += "## 最佳实践\n\n"
            for bp in analysis["best_practices"]:
                content += f"### {bp.get('practice', 'Practice')}\n\n"
                content += f"**原因:**\n{bp.get('reason', '')}\n\n"

        content += "\n---\n\n[返回目录](index.md)"
        return content

    def _generate_walkthrough_page(self, cw, prep_res):
        walkthrough = cw.get("walkthrough", {})
        file_path = cw["file_path"]

        content = f"# 代码走读: {file_path}\n\n"

        content += "## 文件概述\n\n"
        content += walkthrough.get("file_overview", "") + "\n\n"

        content += "## 执行流程\n\n"
        content += walkthrough.get("execution_flow", "") + "\n\n"
        content += walkthrough.get("execution_flow_diagram", "") + "\n\n"

        if walkthrough.get("key_variables"):
            content += "## 关键变量\n\n"
            content += "| 变量名 | 含义 | 设计理由 |\n"
            content += "|--------|------|----------|\n"
            for var in walkthrough["key_variables"]:
                content += f"| {var.get('name', '')} | {var.get('meaning', '')[:50]} | {var.get('design_reason', '')[:50]} |\n"
            content += "\n"

        if walkthrough.get("design_decisions"):
            content += "## 重要设计决策\n\n"
            for decision in walkthrough["design_decisions"]:
                content += f"### {decision.get('decision', 'Decision')[:50]}\n\n"
                content += f"**理由:**\n{decision.get('rationale', '')}\n\n"
                content += f"**权衡:**\n{decision.get('trade_offs', '')}\n\n"

        if walkthrough.get("edge_cases"):
            content += "## 边界情况处理\n\n"
            for ec in walkthrough["edge_cases"]:
                content += f"### {ec.get('case', 'Case')}\n\n"
                content += f"**处理方式:**\n{ec.get('handling', '')}\n\n"

        if walkthrough.get("section_walkthrough"):
            content += "## 逐段详解\n\n"
            for section in walkthrough["section_walkthrough"]:
                content += f"### {section.get('section', 'Section')}\n\n"
                if section.get("code_snippet"):
                    content += f"```python\n{section['code_snippet']}\n```\n\n"
                content += section.get("explanation", "") + "\n\n"

        content += "\n---\n\n[返回目录](index.md)"
        return content

    def _generate_synthesis_page(self, prep_res):
        synthesis = prep_res["tutorial_synthesis"]
        project_name = prep_res["project_name"]

        content = f"# 快速入门: {project_name}\n\n"

        content += "## 快速开始\n\n"
        content += synthesis.get("quick_start", "") + "\n\n"

        if synthesis.get("learning_paths"):
            content += "## 学习路径\n\n"
            for level, path in synthesis["learning_paths"].items():
                content += f"### {path.get('description', level)}\n\n"
                content += "推荐章节:\n"
                for ch_idx in path.get("chapters", []):
                    if 0 <= ch_idx < len(prep_res["abstractions"]):
                        content += f"- {prep_res['abstractions'][ch_idx]['name']}\n"
                content += "\n"

        content += "## 架构全景\n\n"
        content += synthesis.get("architecture_summary", "") + "\n\n"

        if synthesis.get("pattern_quick_ref"):
            content += "## 设计模式速查\n\n"
            content += "| 模式 | 使用位置 | 选择原因 |\n"
            content += "|------|----------|----------|\n"
            for p in synthesis["pattern_quick_ref"]:
                content += f"| {p.get('pattern', '')} | {p.get('location', '')} | {p.get('reason', '')} |\n"
            content += "\n"

        if synthesis.get("faq"):
            content += "## 常见问题\n\n"
            for faq in synthesis["faq"]:
                content += f"### {faq.get('question', 'Q')}\n\n"
                content += f"{faq.get('answer', '')}\n\n"

        if synthesis.get("glossary"):
            content += "## 术语表\n\n"
            content += "| 术语 | 定义 |\n"
            content += "|------|------|\n"
            for term in synthesis["glossary"]:
                content += f"| {term.get('term', '')} | {term.get('definition', '')} |\n"
            content += "\n"

        content += synthesis.get("learning_path_diagram", "") + "\n\n"

        content += "\n---\n\n[返回目录](index.md)"
        return content

    def post(self, shared, prep_res, exec_res):
        shared["final_output_dir"] = exec_res
        print(f"\nEnhanced tutorial generation complete! Files are in: {exec_res}")
