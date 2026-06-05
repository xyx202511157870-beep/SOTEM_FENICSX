from sotem_ip import FiniteWireSurvey, LayerModel


def main():
    survey = FiniteWireSurvey()
    survey.validate(expected_length=1000.0, expected_offset=500.0)
    layers = LayerModel()
    print(f"source length: {survey.source_length:g} m")
    print(f"parallel offset: {survey.parallel_offset:g} m")
    print(f"empymod depth/res: {layers.empymod_depth_res()}")


if __name__ == "__main__":
    main()

