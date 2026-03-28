# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenMed model loader implementation
"""

from typing import Optional

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    ModelConfig,
    StrEnum,
)
from ...base import ForgeModel

_VARIANT_SAMPLE_TEXTS = {
    "ZeroShot-NER-Species-Small-166M": "Escherichia coli and Staphylococcus aureus were isolated from the patient samples.",
    "ZeroShot-NER-DNA-Medium-209M": "The p53 protein plays a crucial role in tumor suppression.",
    "ZeroShot-NER-Species-Medium-209M": "Escherichia coli bacteria were found in the water samples alongside Staphylococcus aureus.",
}

_VARIANT_LABELS = {
    "ZeroShot-NER-Species-Small-166M": ["SPECIES"],
    "ZeroShot-NER-DNA-Medium-209M": ["DNA", "RNA", "cell_line", "cell_type", "protein"],
    "ZeroShot-NER-Species-Medium-209M": ["SPECIES"],
}


class ModelVariant(StrEnum):
    OPENMED_ZEROSHOT_NER_DISEASE_MULTI = "ZeroShot-NER-Disease-Multi-209M"
    OPENMED_ZEROSHOT_NER_PATHOLOGY_MEDIUM = "ZeroShot-NER-Pathology-Medium-209M"
    OPENMED_ZEROSHOT_NER_PHARMA_TINY = "ZeroShot-NER-Pharma-Tiny-60M"
    OPENMED_ZEROSHOT_NER_SPECIES_SMALL = "ZeroShot-NER-Species-Small-166M"
    OPENMED_ZEROSHOT_NER_DNA_MEDIUM = "ZeroShot-NER-DNA-Medium-209M"
    OPENMED_ZEROSHOT_NER_SPECIES_MEDIUM = "ZeroShot-NER-Species-Medium-209M"


class ModelLoader(ForgeModel):
    """OpenMed model loader implementation."""

    _VARIANTS = {
        ModelVariant.OPENMED_ZEROSHOT_NER_DISEASE_MULTI: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-ZeroShot-NER-Disease-Multi-209M"
        ),
        ModelVariant.OPENMED_ZEROSHOT_NER_PATHOLOGY_MEDIUM: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-ZeroShot-NER-Pathology-Medium-209M"
        ),
        ModelVariant.OPENMED_ZEROSHOT_NER_PHARMA_TINY: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-ZeroShot-NER-Pharma-Tiny-60M"
        ),
        ModelVariant.OPENMED_ZEROSHOT_NER_SPECIES_SMALL: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-ZeroShot-NER-Species-Small-166M"
        ),
        ModelVariant.OPENMED_ZEROSHOT_NER_DNA_MEDIUM: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-ZeroShot-NER-DNA-Medium-209M"
        ),
        ModelVariant.OPENMED_ZEROSHOT_NER_SPECIES_MEDIUM: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-ZeroShot-NER-Species-Medium-209M"
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPENMED_ZEROSHOT_NER_ANATOMY_SMALL

    _SAMPLE_TEXTS = {
        ModelVariant.OPENMED_ZEROSHOT_NER_SPECIES_SMALL: "Escherichia coli and Staphylococcus aureus were isolated from the patient samples.",
        ModelVariant.OPENMED_ZEROSHOT_NER_PROTEIN_LARGE: "The Maillard reaction is responsible for the browning of many foods.",
        ModelVariant.OPENMED_ZEROSHOT_NER_ONCOLOGY_BASE: "Mutations in KRAS gene drive oncogenic transformation in colorectal cancer cells.",
        ModelVariant.OPENMED_ZEROSHOT_NER_GENOME_SMALL: "The EGFR gene mutation was identified in lung cancer patients.",
        ModelVariant.OPENMED_ZEROSHOT_NER_GENOME_LARGE: "The BRCA1 and TP53 genes play critical roles in tumor suppression pathways.",
        ModelVariant.OPENMED_ZEROSHOT_NER_ONCOLOGY_MULTI: "Mutations in KRAS gene drive oncogenic transformation in colorectal cancer cells.",
        ModelVariant.OPENMED_ZEROSHOT_NER_PATHOLOGY_XLARGE: "The biopsy revealed adenocarcinoma with lymphovascular invasion in the resected colon specimen.",
    }

    _LABELS = {
        ModelVariant.OPENMED_ZEROSHOT_NER_SPECIES_SMALL: ["SPECIES"],
        ModelVariant.OPENMED_ZEROSHOT_NER_PROTEIN_LARGE: [
            "protein",
            "protein_complex",
            "protein_enum",
            "protein_family_or_group",
            "protein_variant",
        ],
        ModelVariant.OPENMED_ZEROSHOT_NER_ONCOLOGY_BASE: [
            "Cancer",
            "Gene_or_gene_product",
            "Cell",
            "Simple_chemical",
            "Tissue",
        ],
        ModelVariant.OPENMED_ZEROSHOT_NER_GENOME_SMALL: ["GENE/PROTEIN"],
        ModelVariant.OPENMED_ZEROSHOT_NER_GENOME_LARGE: ["GENE/PROTEIN"],
        ModelVariant.OPENMED_ZEROSHOT_NER_ONCOLOGY_MULTI: [
            "Amino_acid",
            "Anatomical_system",
            "Cancer",
            "Cell",
            "Cellular_component",
            "Developing_anatomical_structure",
            "Gene_or_gene_product",
            "Immaterial_anatomical_entity",
            "Multi-tissue_structure",
            "Organ",
            "Organism",
            "Organism_subdivision",
            "Organism_substance",
            "Pathological_formation",
            "Simple_chemical",
            "Tissue",
        ],
        ModelVariant.OPENMED_ZEROSHOT_NER_PATHOLOGY_XLARGE: ["DISEASE"],
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="OpenMed",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the OpenMed GLiNER model."""
        from gliner import GLiNER

        model_name = self._variant_config.pretrained_model_name
        model = GLiNER.from_pretrained(model_name, **kwargs)
        self.model = model
        return self.model.eval()

    _VARIANT_INPUTS = {
        ModelVariant.OPENMED_ZEROSHOT_NER_SPECIES_SMALL: {
            "text": "Escherichia coli and Staphylococcus aureus were isolated from the patient samples.",
            "labels": ["SPECIES"],
        },
        ModelVariant.OPENMED_ZEROSHOT_NER_SPECIES_BASE: {
            "text": "Escherichia coli and Staphylococcus aureus were isolated from the patient samples.",
            "labels": ["SPECIES"],
        },
        ModelVariant.OPENMED_ZEROSHOT_NER_ONCOLOGY_LARGE: {
            "text": "Mutations in KRAS gene drive oncogenic transformation in colorectal cancer cells.",
            "labels": [
                "Gene_or_gene_product",
                "Cancer",
                "Cell",
                "Simple_chemical",
                "Organ",
                "Tissue",
                "Organism",
            ],
        },
        ModelVariant.OPENMED_ZEROSHOT_NER_BLOODCANCER_LARGE: {
            "text": "The patient presented with chronic lymphocytic leukemia symptoms.",
            "labels": ["CL"],
        },
        ModelVariant.OPENMED_ZEROSHOT_NER_DNA_BASE: {
            "text": "The p53 protein plays a crucial role in tumor suppression.",
            "labels": ["DNA", "RNA", "cell_line", "cell_type", "protein"],
        },
        ModelVariant.OPENMED_ZEROSHOT_NER_ORGANISM_MULTI: {
            "text": "Caenorhabditis elegans is a model organism for genetic studies.",
            "labels": ["SPECIES"],
        },
        ModelVariant.OPENMED_ZEROSHOT_NER_PATHOLOGY_TINY: {
            "text": "Early detection of breast cancer improves survival rates.",
            "labels": ["DISEASE"],
        },
    }

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the OpenMed model.

        Returns a batch suitable for the GLiNER model forward pass.
        """
        text = _VARIANT_SAMPLE_TEXTS[self._variant_name]
        self.text = [text]
        labels = _VARIANT_LABELS[self._variant_name]
        entity_types = list(dict.fromkeys(labels))

        (
            tokens,
            all_start_token_idx_to_text_idx,
            all_end_token_idx_to_text_idx,
        ) = self.model.prepare_inputs(
            texts=[text],
        )
        self.all_start_token_idx_to_text_idx = all_start_token_idx_to_text_idx
        self.all_end_token_idx_to_text_idx = all_end_token_idx_to_text_idx

        input_x = self.model.prepare_base_input(tokens)

        collator = self.model.data_collator_class(
            self.model.config,
            data_processor=self.model.data_processor,
            return_tokens=True,
            return_entities=True,
            return_id_to_classes=True,
            prepare_labels=False,
        )

        batch = collator(input_x, entity_types=entity_types)
        self.batch = batch
        return batch

    def post_processing(self, co_out):
        decoded = self.model.decoder.decode(
            self.batch["tokens"],
            self.batch["id_to_classes"],
            co_out,
            flat_ner=True,
            threshold=0.5,
            multi_label=False,
        )
        all_entities = []
        for i, spans in enumerate(decoded):
            start_token_idx_to_text_idx = self.all_start_token_idx_to_text_idx[i]
            end_token_idx_to_text_idx = self.all_end_token_idx_to_text_idx[i]
            entities = []
            for span in spans:
                start_text_idx = start_token_idx_to_text_idx[span.start]
                end_text_idx = end_token_idx_to_text_idx[span.end]
                ent_details = {
                    "start": start_text_idx,
                    "end": end_text_idx,
                    "text": self.text[i][start_text_idx:end_text_idx],
                    "label": span.entity_type,
                    "score": span.score,
                }
                entities.append(ent_details)
            all_entities.append(entities)
        return all_entities
