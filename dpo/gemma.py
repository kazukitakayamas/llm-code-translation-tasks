def custom(cfg, **kwargs):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    For My Customized Data
    """

    def transform_fn(sample):
        sample[
            "prompt"
        ] = f"<bos><start_of_turn>user\n{sample['prompt']}<end_of_turn>\n<start_of_turn>model\n"
        sample["chosen"] = f"{sample['chosen']}<end_of_turn>"
        sample["rejected"] = f"{sample['rejected']}<end_of_turn>"
        return sample

    return transform_fn