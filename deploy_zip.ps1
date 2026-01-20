$ver="1.0.0"
$src="dist\MmEmojiManager"
$zip="MmEmojiManager_v$ver`_win64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path "$src\*" -DestinationPath $zip
