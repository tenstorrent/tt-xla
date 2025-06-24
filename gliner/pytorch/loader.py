# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLiNER model loader implementation
"""


from gliner import GLiNER
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """GLiNER model loader implementation."""

    # Shared configuration parameters
    model_name = "urchade/gliner_largev2"

    @classmethod
    def load_model(cls):
        """Load and return the GLiNER model instance with default settings.

        Returns:
            torch.nn.Module: The GLiNER model instance.
        """
        model = GLiNER.from_pretrained(cls.model_name)
        return model.batch_predict_entities

    @classmethod
    def load_inputs(cls, batch_size=1):
        """Load and return sample inputs for the GLiNER model with default settings.

        Args:
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """
        text = """
        Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃ'tjɐnu ʁɔ'naldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
        """

        labels = ["person", "award", "date", "competitions", "teams"]

        # Batch the inputs using list replication
        texts = [text] * batch_size

        return (texts, labels)
