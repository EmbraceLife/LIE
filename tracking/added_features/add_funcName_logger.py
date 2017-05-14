# replace the following code with its corresponding part in __main__.main(args)

	config(
		level=loglevel.get(args.verbose, logging.TRACE),
		format='{color}[%(levelname)s %(asctime)s %(name)s %(funcName)s:%(lineno)s]{reset} '
			'%(message)s'.format(
				color='' if args.no_color else '$COLOR',
				reset='' if args.no_color else '$RESET'
			)
	)
