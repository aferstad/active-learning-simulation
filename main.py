if __name__ == '__main__':  # to avoid multiprocessor children to begin from start

    from alsRepeaterLauncher import AlsRepeaterLauncher
    import alsDataManager
    import sys  # to get arguments from terminal

    input_arguments = sys.argv

    params = alsDataManager.open_dict_from_json(input_arguments[1])
    print(params)
    launcher = AlsRepeaterLauncher(params)

    results = launcher.run_3_dimensional_varied_reps()

    save_path = 'output/jsons/' + input_arguments[1] + '.txt'
    alsDataManager.save_dict_as_json(output_dict=results, output_path=save_path)