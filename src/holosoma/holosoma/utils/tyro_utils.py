import tyro
import tyro.conf

TYRO_CONIFG = (tyro.conf.CascadeSubcommandArgs, tyro.conf.FlagConversionOff, tyro.conf.UsePythonSyntaxForLiteralCollections) #tyro命令行格式相关设置
