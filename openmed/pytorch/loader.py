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


class ModelVariant(StrEnum):
    OPENMED_ZEROSHOT_NER_SPECIES_TINY = "ZeroShot-NER-Species-Tiny-60M"
    OPENMED_ZEROSHOT_NER_SPECIES_SMALL = "ZeroShot-NER-Species-Small-166M"
    OPENMED_ZEROSHOT_NER_PHARMA_LARGE = "ZeroShot-NER-Pharma-Large-459M"
    OPENMED_ZEROSHOT_NER_DISEASE_BASE = "ZeroShot-NER-Disease-Base-220M"
    OPENMED_ZEROSHOT_NER_BLOODCANCER_MULTI = "ZeroShot-NER-BloodCancer-Multi-209M"


class ModelLoader(ForgeModel):
    """OpenMed model loader implementation."""

    _VARIANTS = {
        ModelVariant.OPENMED_ZEROSHOT_NER_SPECIES_TINY: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-ZeroShot-NER-Species-Tiny-60M"
        ),
        ModelVariant.OPENMED_ZEROSHOT_NER_SPECIES_SMALL: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-ZeroShot-NER-Species-Small-166M"
        ),
        ModelVariant.OPENMED_ZEROSHOT_NER_PHARMA_LARGE: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-ZeroShot-NER-Pharma-Large-459M"
        ),
        ModelVariant.OPENMED_ZEROSHOT_NER_DISEASE_BASE: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-ZeroShot-NER-Disease-Base-220M"
        ),
        ModelVariant.OPENMED_ZEROSHOT_NER_BLOODCANCER_MULTI: ModelConfig(
            pretrained_model_name="OpenMed/OpenMed-ZeroShot-NER-BloodCancer-Multi-209M"
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

    _VARIANT_SAMPLES = {
        ModelVariant.OPENMED_ZEROSHOT_NER_SPECIES_SMALL: {
            "text": "Escherichia coli and Staphylococcus aureus were isolated from the patient samples.",
            "labels": ["SPECIES"],
        },
        ModelVariant.OPENMED_ZEROSHOT_NER_DISEASE_XLARGE: {
            "text": "The patient was diagnosed with diabetes mellitus type 2 and hypertension.",
            "labels": ["DISEASE"],
        },
        ModelVariant.OPENMED_ZEROSHOT_NER_GENOMIC_SMALL: {
            "text": "The BRCA2 gene is associated with hereditary breast cancer.",
            "labels": ["Cell-line-name"],
        },
        ModelVariant.OPENMED_ZEROSHOT_NER_CHEMICAL_XLARGE: {
            "text": "The patient was administered acetylsalicylic acid for pain relief.",
            "labels": ["CHEM"],
        },
    }

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the OpenMed model.

        Returns a batch suitable for the GLiNER model forward pass.
        """
        if self._variant == ModelVariant.OPENMED_ZEROSHOT_NER_PHARMA_LARGE:
            text = "Administration of metformin reduced glucose levels significantly."
            labels = ["CHE"]
        elif self._variant == ModelVariant.OPENMED_ZEROSHOT_NER_DISEASE_BASE:
            text = "The patient was diagnosed with diabetes mellitus type 2."
            labels = ["DISEASE"]
        elif self._variant == ModelVariant.OPENMED_ZEROSHOT_NER_BLOODCANCER_MULTI:
            text = "The patient presented with chronic lymphocytic leukemia symptoms."
            labels = ["CL"]
        else:
            text = "Escherichia coli and Staphylococcus aureus were isolated from the patient samples."
            labels = ["SPECIES"]
        self.text = [text]
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
