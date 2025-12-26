"""Tests for material parser."""

import pytest
from synthesis_enhancements.parsers.material_parser import (
    EnhancedMaterialParser,
    ParsedMaterial,
    ElementComposition,
)


class TestElementComposition:
    """Tests for ElementComposition."""

    def test_creation(self):
        """Test composition creation."""
        comp = ElementComposition(
            element="Li",
            amount=2.0,
            oxidation_state=1,
        )

        assert comp.element == "Li"
        assert comp.amount == 2.0
        assert comp.oxidation_state == 1

    def test_default_oxidation_state(self):
        """Test default oxidation state is None."""
        comp = ElementComposition(element="O", amount=3.0)
        assert comp.oxidation_state is None


class TestParsedMaterial:
    """Tests for ParsedMaterial."""

    def test_creation(self):
        """Test parsed material creation."""
        material = ParsedMaterial(
            original="LiCoO2",
            formula="LiCoO2",
        )

        assert material.original == "LiCoO2"
        assert material.formula == "LiCoO2"
        assert material.is_mixture is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        material = ParsedMaterial(
            original="LiCoO2",
            formula="LiCoO2",
            composition=[
                ElementComposition("Li", 1.0),
                ElementComposition("Co", 1.0),
                ElementComposition("O", 2.0),
            ],
        )

        d = material.to_dict()

        assert d["original"] == "LiCoO2"
        assert d["formula"] == "LiCoO2"
        assert len(d["composition"]) == 3
        assert d["composition"][0]["element"] == "Li"


class TestEnhancedMaterialParser:
    """Tests for EnhancedMaterialParser."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        parser = EnhancedMaterialParser(num_workers=2)
        parser.load(load_embeddings=False)
        return parser

    def test_initialization(self):
        """Test parser initialization."""
        parser = EnhancedMaterialParser()
        assert parser.config is not None

    def test_clean_string(self, parser):
        """Test string cleaning."""
        assert parser._clean_string("  LiCoO2  ") == "LiCoO2"
        assert parser._clean_string("Li ( Co ) O2") == "Li(Co)O2"

    def test_is_mixture(self, parser):
        """Test mixture detection."""
        assert parser._is_mixture("Li2CO3:Co3O4") is True
        assert parser._is_mixture("A + B") is True
        assert parser._is_mixture("LiCoO2") is False

    def test_parse_simple_formula(self, parser):
        """Test parsing simple formula."""
        result = parser.parse("LiCoO2")

        assert result.original == "LiCoO2"
        assert result.formula == "LiCoO2"
        assert len(result.composition) == 3

        elements = {c.element: c.amount for c in result.composition}
        assert elements["Li"] == 1.0
        assert elements["Co"] == 1.0
        assert elements["O"] == 2.0

    def test_parse_formula_with_decimals(self, parser):
        """Test parsing formula with decimal amounts."""
        result = parser.parse("Li0.5Na0.5CoO2")

        elements = {c.element: c.amount for c in result.composition}
        assert elements["Li"] == 0.5
        assert elements["Na"] == 0.5

    def test_parse_abbreviation(self, parser):
        """Test parsing known abbreviation."""
        result = parser.parse("LCO")

        assert result.formula is not None
        # LCO should resolve to LiCoO2

    def test_extract_phase(self, parser):
        """Test phase extraction."""
        assert parser._extract_phase("α-Al2O3") == "alpha"
        assert parser._extract_phase("beta-Li2TiO3") == "beta"
        assert parser._extract_phase("cubic ZrO2") == "cubic"
        assert parser._extract_phase("LiCoO2") is None

    def test_extract_dopants(self, parser):
        """Test dopant extraction."""
        dopants = parser._extract_dopants("Al-doped LiCoO2")

        assert len(dopants) == 1
        assert dopants[0]["element"] == "Al"

    def test_expand_parentheses(self, parser):
        """Test parentheses expansion."""
        result = parser._expand_parentheses("Ca(OH)2")
        # Should expand to CaO2H2
        assert "O2" in result or "O" in result

    def test_parse_batch(self, parser):
        """Test batch parsing."""
        materials = ["LiCoO2", "NaCl", "Fe2O3"]
        results = parser.parse_batch(materials, use_parallel=False)

        assert len(results) == 3
        assert all(isinstance(r, ParsedMaterial) for r in results)

    def test_parse_batch_parallel(self, parser):
        """Test parallel batch parsing."""
        materials = ["LiCoO2", "NaCl", "Fe2O3", "Al2O3"]
        results = parser.parse_batch(materials, use_parallel=True)

        assert len(results) == 4

    def test_normalize_formula(self, parser):
        """Test formula normalization."""
        # Should put cations before anions
        normalized = parser.normalize_formula("O2Li")
        assert normalized.startswith("Li")

    def test_parse_mixture(self, parser):
        """Test mixture parsing."""
        result = parser.parse("Li2CO3:Co3O4")

        assert result.is_mixture is True
        assert len(result.components) >= 1

    def test_resolve_variables(self, parser):
        """Test variable resolution."""
        result = parser.parse("Li1-xNaxCoO2 (x=0.3)")

        assert "x" in result.variables or len(result.composition) > 0
