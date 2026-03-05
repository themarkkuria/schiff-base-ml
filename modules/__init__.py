"""
Molecular design modules.
"""

from .amp_module import (
    AMPGenerator,
    AMPFeaturizer, 
    AMPModel,
    AMPRanker,
    generate_synthetic_amp_data
)

from .repurposed_module import (
    RepurposedDrugFetcher,
    RepurposedFeaturizer,
    ScaffoldSimilarity,
    RepurposedModel,
    DrugRepurposingPipeline,
    generate_synthetic_repurposing_data
)

from .polyphenols_module import (
    PolyphenolGenerator,
    PolyphenolFeaturizer,
    CoCrystalPredictor,
    PolyphenolModel,
    PolyphenolPipeline,
    generate_synthetic_polyphenol_data,
    POLYPHENOL_STARTING_POINTS
)

from .protac_module import (
    PROTACGenerator,
    PROTACFeaturizer,
    PROTACModel,
    PROTACPipeline,
    generate_synthetic_protac_data,
    PROTEASE_TARGETS,
    LINKER_TYPES
)

__all__ = [
    # AMP Module
    'AMPGenerator',
    'AMPFeaturizer', 
    'AMPModel',
    'AMPRanker',
    'generate_synthetic_amp_data',
    # Repurposed Module
    'RepurposedDrugFetcher',
    'RepurposedFeaturizer',
    'ScaffoldSimilarity',
    'RepurposedModel',
    'DrugRepurposingPipeline',
    'generate_synthetic_repurposing_data',
    # Polyphenols Module
    'PolyphenolGenerator',
    'PolyphenolFeaturizer',
    'CoCrystalPredictor',
    'PolyphenolModel',
    'PolyphenolPipeline',
    'generate_synthetic_polyphenol_data',
    'POLYPHENOL_STARTING_POINTS',
    # PROTAC Module
    'PROTACGenerator',
    'PROTACFeaturizer',
    'PROTACModel',
    'PROTACPipeline',
    'generate_synthetic_protac_data',
    'PROTASE_TARGETS',
    'LINKER_TYPES'
]
