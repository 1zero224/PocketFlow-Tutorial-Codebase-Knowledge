"""
Enhanced tutorial generation flow with deep analysis capabilities.

This module provides an extended pipeline that generates much more detailed
and comprehensive documentation compared to the standard flow.

Pipeline:
FetchRepo → IdentifyAbstractions → DeepAnalysisNode → DesignPatternNode
→ ArchitectureOverviewNode → AnalyzeRelationships → OrderChapters
→ CodeWalkthroughNode → EnhancedWriteChaptersNode → TutorialSynthesisNode
→ DeepCombineTutorial
"""

from pocketflow import Flow
from nodes import (
    FetchRepo,
    IdentifyAbstractions,
    AnalyzeRelationships,
    OrderChapters,
)
from deep_nodes import (
    DeepAnalysisNode,
    DesignPatternNode,
    ArchitectureOverviewNode,
    CodeWalkthroughNode,
    EnhancedWriteChaptersNode,
    TutorialSynthesisNode,
    DeepCombineTutorial,
)


def create_deep_tutorial_flow():
    """Creates and returns the enhanced deep tutorial generation flow.

    This flow adds the following capabilities over the standard flow:
    - Deep analysis of each abstraction (design motivation, trade-offs, etc.)
    - Design pattern identification and analysis
    - Comprehensive architecture overview with multiple Mermaid diagrams
    - Detailed code walkthroughs for key files
    - Enhanced chapters with 10x more detail
    - Final synthesis with quick start guide, FAQ, glossary, etc.

    Pipeline:
    FetchRepo → IdentifyAbstractions → DeepAnalysisNode → DesignPatternNode
    → ArchitectureOverviewNode → AnalyzeRelationships → OrderChapters
    → CodeWalkthroughNode → EnhancedWriteChaptersNode → TutorialSynthesisNode
    → DeepCombineTutorial
    """

    # Instantiate nodes
    fetch_repo = FetchRepo()
    identify_abstractions = IdentifyAbstractions(max_retries=5, wait=20)

    # New deep analysis nodes
    deep_analysis = DeepAnalysisNode(max_retries=5, wait=20)
    design_patterns = DesignPatternNode(max_retries=5, wait=20)
    architecture_overview = ArchitectureOverviewNode(max_retries=5, wait=20)

    # Standard nodes (reused)
    analyze_relationships = AnalyzeRelationships(max_retries=5, wait=20)
    order_chapters = OrderChapters(max_retries=5, wait=20)

    # New enhanced nodes
    code_walkthrough = CodeWalkthroughNode(max_retries=5, wait=20)
    enhanced_write_chapters = EnhancedWriteChaptersNode(max_retries=5, wait=20)
    tutorial_synthesis = TutorialSynthesisNode(max_retries=5, wait=20)
    deep_combine = DeepCombineTutorial()

    # Connect nodes in sequence using >> operator
    # Each node has exactly one default successor
    fetch_repo >> identify_abstractions
    identify_abstractions >> deep_analysis
    deep_analysis >> design_patterns
    design_patterns >> architecture_overview
    architecture_overview >> analyze_relationships
    analyze_relationships >> order_chapters
    order_chapters >> code_walkthrough
    code_walkthrough >> enhanced_write_chapters
    enhanced_write_chapters >> tutorial_synthesis
    tutorial_synthesis >> deep_combine

    # Create the flow starting with FetchRepo
    deep_flow = Flow(start=fetch_repo)

    return deep_flow


