"""Enhanced material parsing using transformer models and parallel processing."""

from dataclasses import dataclass, field
from typing import Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import logging
import re
import json

from tqdm import tqdm

from synthesis_enhancements.models import (
    TransformerModelFactory,
    SynthesisEmbeddingModel,
    ModelConfig,
)
from synthesis_enhancements.processing import (
    BatchProcessor,
    ParallelProcessor,
)
from synthesis_enhancements.utils.config import ParserConfig, ProcessingConfig

logger = logging.getLogger(__name__)


@dataclass
class ElementComposition:
    """Elemental composition of a material.

    Attributes:
        element: Element symbol
        amount: Stoichiometric amount
        oxidation_state: Oxidation state (if known)
    """
    element: str
    amount: float
    oxidation_state: Optional[int] = None


@dataclass
class ParsedMaterial:
    """Result of parsing a material string.

    Attributes:
        original: Original material string
        formula: Normalized formula
        composition: List of element compositions
        is_mixture: Whether material is a mixture
        components: Components if mixture
        dopants: Identified dopants
        variables: Unresolved variables
        phase: Crystal phase (if identified)
        confidence: Parsing confidence
        metadata: Additional metadata
    """
    original: str
    formula: Optional[str] = None
    composition: list[ElementComposition] = field(default_factory=list)
    is_mixture: bool = False
    components: list["ParsedMaterial"] = field(default_factory=list)
    dopants: list[dict[str, Any]] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    phase: Optional[str] = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original": self.original,
            "formula": self.formula,
            "composition": [
                {"element": c.element, "amount": c.amount, "oxidation_state": c.oxidation_state}
                for c in self.composition
            ],
            "is_mixture": self.is_mixture,
            "components": [c.to_dict() for c in self.components],
            "dopants": self.dopants,
            "variables": self.variables,
            "phase": self.phase,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class EnhancedMaterialParser:
    """Enhanced material parser with transformer-based understanding.

    Parses material strings into structured compositions with support
    for complex formulas, mixtures, dopants, and variables. Uses
    transformer models for improved parsing of ambiguous cases.

    Example:
        >>> parser = EnhancedMaterialParser(num_workers=4)
        >>> parser.load()
        >>>
        >>> result = parser.parse("Li0.5Na0.5CoO2")
        >>> print(result.composition)
        >>>
        >>> results = parser.parse_batch([
        ...     "LiCoO2",
        ...     "Li2CO3:Co3O4 (1:1 molar ratio)",
        ...     "La1-xSrxMnO3 (x=0.3)"
        ... ])

    Attributes:
        config: Parser configuration
        embedding_model: Embedding model for similarity matching
    """

    # Periodic table data
    ELEMENTS = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
        'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
        'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
        'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
        'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
        'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
        'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
        'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
        'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
    }

    # Parsing patterns
    FORMULA_PATTERN = re.compile(
        r"([A-Z][a-z]?)(\d*\.?\d*)"
    )

    MIXTURE_PATTERNS = [
        re.compile(r"(.+?)\s*[:/]\s*(.+?)(?:\s*\((\d+:\d+)\))?"),  # A:B (1:1)
        re.compile(r"(.+?)\s*\+\s*(.+)"),  # A + B
        re.compile(r"(\d+\.?\d*)\s*(.+?)\s*[-–]\s*(\d+\.?\d*)\s*(.+)"),  # 0.5A-0.5B
    ]

    DOPANT_PATTERN = re.compile(
        r"([A-Z][a-z]?)\s*[-–]?\s*doped",
        re.IGNORECASE,
    )

    VARIABLE_PATTERN = re.compile(
        r"([A-Z][a-z]?)(\d*-)?([xyz])(\d*-)?([A-Z][a-z]?)?\s*=?\s*(\d*\.?\d*)?",
        re.IGNORECASE,
    )

    PHASE_PATTERNS = {
        "alpha": re.compile(r"α|alpha", re.I),
        "beta": re.compile(r"β|beta", re.I),
        "gamma": re.compile(r"γ|gamma", re.I),
        "delta": re.compile(r"δ|delta", re.I),
        "cubic": re.compile(r"cubic", re.I),
        "tetragonal": re.compile(r"tetragonal", re.I),
        "hexagonal": re.compile(r"hexagonal", re.I),
        "orthorhombic": re.compile(r"orthorhombic", re.I),
        "monoclinic": re.compile(r"monoclinic", re.I),
        "triclinic": re.compile(r"triclinic", re.I),
        "rhombohedral": re.compile(r"rhombohedral", re.I),
    }

    def __init__(
        self,
        model_name: str = "matbert",
        num_workers: int = 4,
        batch_size: int = 64,
        config: Optional[ParserConfig] = None,
    ) -> None:
        """Initialize the material parser.

        Args:
            model_name: Name of transformer model for embeddings
            num_workers: Number of parallel workers
            batch_size: Batch size for processing
            config: Parser configuration
        """
        self.config = config or ParserConfig(
            model_config=ModelConfig(model_name=model_name),
            processing_config=ProcessingConfig(
                num_workers=num_workers,
                batch_size=batch_size,
            ),
        )

        self._embedding_model: Optional[SynthesisEmbeddingModel] = None
        self._abbreviations: dict[str, str] = {}
        self._is_loaded = False

    def load(
        self,
        load_embeddings: bool = True,
        abbreviations_path: Optional[str] = None,
    ) -> "EnhancedMaterialParser":
        """Load the parser resources.

        Args:
            load_embeddings: Whether to load embedding model
            abbreviations_path: Path to abbreviations dictionary

        Returns:
            Self for method chaining
        """
        if self._is_loaded:
            logger.warning("Parser already loaded")
            return self

        logger.info("Loading material parser...")

        # Load embedding model for similarity matching
        if load_embeddings:
            self._embedding_model = TransformerModelFactory.create_embedding_model(
                model_name=self.config.model_config.model_name,
            )
            self._embedding_model.load()

        # Load abbreviations
        if abbreviations_path:
            with open(abbreviations_path) as f:
                self._abbreviations = json.load(f)
        else:
            # Default abbreviations
            self._abbreviations = {
                "LLZO": "Li7La3Zr2O12",
                "LNMO": "LiNi0.5Mn1.5O4",
                "NMC": "LiNixMnyCozO2",
                "LFP": "LiFePO4",
                "LCO": "LiCoO2",
                "NCA": "LiNixCoyAlzO2",
                "YSZ": "Y2O3-ZrO2",
                "BSCF": "BaxSr1-xCoyFe1-yO3",
                "LSCF": "LaxSr1-xCoyFe1-yO3",
            }

        self._is_loaded = True
        logger.info("Material parser loaded")
        return self

    def parse(
        self,
        material_string: str,
        resolve_variables: bool = True,
    ) -> ParsedMaterial:
        """Parse a material string.

        Args:
            material_string: Material string to parse
            resolve_variables: Whether to resolve variable compositions

        Returns:
            Parsed material result
        """
        # Clean input
        material_string = self._clean_string(material_string)

        # Check for abbreviations
        if material_string.upper() in self._abbreviations:
            material_string = self._abbreviations[material_string.upper()]

        # Initialize result
        result = ParsedMaterial(original=material_string)

        # Check for mixture
        if self._is_mixture(material_string):
            return self._parse_mixture(material_string, resolve_variables)

        # Extract phase information
        result.phase = self._extract_phase(material_string)

        # Extract dopants
        result.dopants = self._extract_dopants(material_string)

        # Parse formula
        formula_str = self._extract_formula(material_string)
        result.formula = formula_str

        # Parse composition
        composition, variables = self._parse_composition(formula_str)
        result.composition = composition
        result.variables = variables

        # Resolve variables if requested
        if resolve_variables and variables:
            result = self._resolve_variables(result, material_string)

        return result

    def parse_batch(
        self,
        material_strings: list[str],
        use_parallel: bool = True,
    ) -> list[ParsedMaterial]:
        """Parse multiple material strings.

        Args:
            material_strings: List of material strings
            use_parallel: Whether to use parallel processing

        Returns:
            List of parsed results
        """
        if not material_strings:
            return []

        if use_parallel and len(material_strings) > 1:
            return self._parse_parallel(material_strings)
        else:
            return self._parse_sequential(material_strings)

    def _parse_sequential(
        self,
        materials: list[str],
    ) -> list[ParsedMaterial]:
        """Parse materials sequentially."""
        results = []
        for mat in tqdm(materials, desc="Parsing materials"):
            try:
                results.append(self.parse(mat))
            except Exception as e:
                logger.warning(f"Failed to parse '{mat}': {e}")
                results.append(ParsedMaterial(original=mat, confidence=0.0))
        return results

    def _parse_parallel(
        self,
        materials: list[str],
    ) -> list[ParsedMaterial]:
        """Parse materials in parallel."""

        def process_fn(mat: str) -> ParsedMaterial:
            try:
                return self.parse(mat)
            except Exception as e:
                logger.warning(f"Failed to parse '{mat}': {e}")
                return ParsedMaterial(original=mat, confidence=0.0)

        processor = ParallelProcessor(
            process_fn=process_fn,
            config=self.config.processing_config,
            name="MaterialParsing",
        )

        results = processor.process(materials)
        return [r.output for r in results]

    def _clean_string(self, s: str) -> str:
        """Clean material string."""
        # Remove extra whitespace
        s = " ".join(s.split())
        # Remove common artifacts
        s = re.sub(r"\s*\(\s*", "(", s)
        s = re.sub(r"\s*\)\s*", ")", s)
        return s.strip()

    def _is_mixture(self, s: str) -> bool:
        """Check if string represents a mixture."""
        for pattern in self.MIXTURE_PATTERNS:
            if pattern.search(s):
                return True
        return False

    def _parse_mixture(
        self,
        material_string: str,
        resolve_variables: bool,
    ) -> ParsedMaterial:
        """Parse a mixture material."""
        result = ParsedMaterial(
            original=material_string,
            is_mixture=True,
        )

        # Try different mixture patterns
        for pattern in self.MIXTURE_PATTERNS:
            match = pattern.match(material_string)
            if match:
                groups = match.groups()
                for g in groups:
                    if g and not re.match(r"^\d+:\d+$", g):
                        component = self.parse(g, resolve_variables)
                        result.components.append(component)
                break

        return result

    def _extract_phase(self, s: str) -> Optional[str]:
        """Extract crystal phase from string."""
        for phase_name, pattern in self.PHASE_PATTERNS.items():
            if pattern.search(s):
                return phase_name
        return None

    def _extract_dopants(self, s: str) -> list[dict[str, Any]]:
        """Extract dopant information."""
        dopants = []
        for match in self.DOPANT_PATTERN.finditer(s):
            dopants.append({
                "element": match.group(1),
                "text": match.group(0),
            })
        return dopants

    def _extract_formula(self, s: str) -> str:
        """Extract formula portion from string."""
        # Remove phase and dopant text
        formula = s
        for pattern in self.PHASE_PATTERNS.values():
            formula = pattern.sub("", formula)
        formula = self.DOPANT_PATTERN.sub("", formula)

        # Remove parenthetical notes
        formula = re.sub(r"\([^()]*\)$", "", formula)

        return formula.strip()

    def _parse_composition(
        self,
        formula: str,
    ) -> tuple[list[ElementComposition], dict[str, Any]]:
        """Parse formula into elemental composition.

        Returns:
            Tuple of (composition list, unresolved variables)
        """
        composition = []
        variables = {}

        # Handle parentheses first
        formula = self._expand_parentheses(formula)

        # Find all element-amount pairs
        for match in self.FORMULA_PATTERN.finditer(formula):
            element = match.group(1)
            amount_str = match.group(2)

            if element not in self.ELEMENTS:
                continue

            # Parse amount
            if not amount_str:
                amount = 1.0
            elif re.match(r"^[\d.]+$", amount_str):
                amount = float(amount_str)
            else:
                # Contains variable
                variables[element] = amount_str
                amount = 0.0  # Placeholder

            composition.append(ElementComposition(
                element=element,
                amount=amount,
            ))

        return composition, variables

    def _expand_parentheses(self, formula: str) -> str:
        """Expand parenthetical groups in formula."""
        # Pattern to match (group)n
        pattern = re.compile(r"\(([^()]+)\)(\d*\.?\d*)")

        while pattern.search(formula):
            def replacer(m):
                group = m.group(1)
                multiplier = float(m.group(2)) if m.group(2) else 1.0

                # Multiply all amounts in group
                result = ""
                for match in self.FORMULA_PATTERN.finditer(group):
                    elem = match.group(1)
                    amt = float(match.group(2)) if match.group(2) else 1.0
                    new_amt = amt * multiplier
                    if new_amt == int(new_amt):
                        new_amt = int(new_amt)
                    result += f"{elem}{new_amt if new_amt != 1 else ''}"

                return result

            formula = pattern.sub(replacer, formula)

        return formula

    def _resolve_variables(
        self,
        result: ParsedMaterial,
        original: str,
    ) -> ParsedMaterial:
        """Resolve variable compositions from context."""
        # Look for explicit values in original string
        # e.g., "x=0.3" or "(x = 0.3)"
        var_value_pattern = re.compile(r"([xyz])\s*=\s*(\d*\.?\d+)")

        for match in var_value_pattern.finditer(original):
            var_name = match.group(1)
            value = float(match.group(2))
            result.variables[var_name] = value

        return result

    def find_similar_materials(
        self,
        material_string: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Find similar materials using embeddings.

        Args:
            material_string: Query material
            candidates: List of candidate materials
            top_k: Number of results to return

        Returns:
            List of (material, similarity) tuples
        """
        if self._embedding_model is None:
            raise RuntimeError("Embedding model not loaded")

        # Embed query
        query_embedding = self._embedding_model.embed(material_string)

        # Embed candidates in batches
        candidate_embeddings = self._embedding_model.batch_embed(
            candidates,
            batch_size=self.config.processing_config.batch_size,
        )

        # Compute similarities
        import numpy as np
        similarities = np.dot(candidate_embeddings, query_embedding)

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            (candidates[i], float(similarities[i]))
            for i in top_indices
        ]

    def normalize_formula(self, formula: str) -> str:
        """Normalize a chemical formula to standard form.

        Args:
            formula: Input formula

        Returns:
            Normalized formula
        """
        result = self.parse(formula)

        if not result.composition:
            return formula

        # Sort elements by electronegativity (simplified: cations then anions)
        cations = []
        anions = []
        for comp in result.composition:
            if comp.element in ['O', 'S', 'N', 'F', 'Cl', 'Br', 'I']:
                anions.append(comp)
            else:
                cations.append(comp)

        # Build normalized formula
        normalized = ""
        for comp in cations + anions:
            amt = comp.amount
            if amt == int(amt):
                amt = int(amt)
            normalized += f"{comp.element}{amt if amt != 1 else ''}"

        return normalized
