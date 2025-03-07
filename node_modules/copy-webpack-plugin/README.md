<div align="center">
  <a href="https://github.com/webpack/webpack">
    <img width="200" height="200"
      src="https://webpack.js.org/assets/icon-square-big.svg">
  </a>
</div>

[![npm][npm]][npm-url]
[![node][node]][node-url]
[![tests][tests]][tests-url]
[![cover][cover]][cover-url]
[![discussion][discussion]][discussion-url]
[![size][size]][size-url]

# copy-webpack-plugin

Copies individual files or entire directories, which already exist, to the build directory.

## Getting Started

To begin, you'll need to install `copy-webpack-plugin`:

```console
npm install copy-webpack-plugin --save-dev
```

or

```console
yarn add -D copy-webpack-plugin
```

or

```console
pnpm add -D copy-webpack-plugin
```

Then add the plugin to your `webpack` config. For example:

**webpack.config.js**

```js
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        { from: "source", to: "dest" },
        { from: "other", to: "public" },
      ],
    }),
  ],
};
```

> **Note**
>
> `copy-webpack-plugin` is not designed to copy files generated from the build process; rather, it is to copy files that already exist in the source tree, as part of the build process.

> **Note**
>
> If you want `webpack-dev-server` to write files to the output directory during development, you can force it with the [`writeToDisk`](https://github.com/webpack/webpack-dev-middleware#writetodisk) option or the [`write-file-webpack-plugin`](https://github.com/gajus/write-file-webpack-plugin).

> **Note**
>
> You can get the original source filename from [Asset Objects](https://webpack.js.org/api/stats/#asset-objects).

## Options

- **[`patterns`](#patterns)**
- **[`options`](#options-1)**

The plugin's signature:

**webpack.config.js**

```js
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        { from: "source", to: "dest" },
        "path/to/source", // absolute or relative, files/directories/globs - see below for examples
      ],
      options: {
        concurrency: 100,
      },
    }),
  ],
};
```

### `Patterns`

- [`from`](#from)
- [`to`](#to)
- [`context`](#context)
- [`globOptions`](#globoptions)
- [`filter`](#filter)
- [`toType`](#totype)
- [`force`](#force)
- [`priority`](#priority)
- [`transform`](#transform)
- [`transformAll`](#transformAll)
- [`noErrorOnMissing`](#noerroronmissing)
- [`info`](#info)

#### `from`

Type:

```ts
type from = string;
```

Default: `undefined`

Glob or path from where we copy files.
Globs accept [fast-glob pattern-syntax](https://github.com/mrmlnc/fast-glob#pattern-syntax).
Glob can only be a `string`.

> **Warning**
>
> Don't use directly `\\` in `from` option if it is a `glob` (i.e `path\to\file.ext`) option because on UNIX the backslash is a valid character inside a path component, i.e., it's not a separator.
> On Windows, the forward slash and the backward slash are both separators.
> Instead please use `/`.

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        "relative/path/to/file.ext",
        "relative/path/to/dir",
        path.resolve(__dirname, "src", "file.ext"),
        path.resolve(__dirname, "src", "dir"),
        "**/*",
        {
          from: "**/*",
        },
        // If absolute path is a `glob` we replace backslashes with forward slashes, because only forward slashes can be used in the `glob`
        path.posix.join(
          path.resolve(__dirname, "src").replace(/\\/g, "/"),
          "*.txt",
        ),
      ],
    }),
  ],
};
```

##### `For windows`

If you define `from` as absolute file path or absolute folder path on `Windows`, you can use windows path segment (`\\`)

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: path.resolve(__dirname, "file.txt"),
        },
      ],
    }),
  ],
};
```

But you should always use forward-slashes in `glob` expressions
See [fast-glob manual](https://github.com/mrmlnc/fast-glob#how-to-write-patterns-on-windows).

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          // If absolute path is a `glob` we replace backslashes with forward slashes, because only forward slashes can be used in the `glob`
          from: path.posix.join(
            path.resolve(__dirname, "fixtures").replace(/\\/g, "/"),
            "*.txt",
          ),
        },
      ],
    }),
  ],
};
```

The `context` behaves differently depending on what the `from` is (`glob`, `file` or `dir`).
More [`examples`](#examples)

#### `to`

Type:

```ts
type to =
  | string
  | ((pathData: { context: string; absoluteFilename?: string }) => string);
```

Default: `compiler.options.output`

##### `string`

Output path.

> **Warning**
>
> Don't use directly `\\` in `to` (i.e `path\to\dest`) option because on UNIX the backslash is a valid character inside a path component, i.e., it's not a separator.
> On Windows, the forward slash and the backward slash are both separators.
> Instead please use `/` or `path` methods.

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "**/*",
          to: "relative/path/to/dest/",
        },
        {
          from: "**/*",
          to: "/absolute/path/to/dest/",
        },
        {
          from: "**/*",
          to: "[path][name].[contenthash][ext]",
        },
      ],
    }),
  ],
};
```

##### `function`

Allows to modify the writing path.

> **Warning**
>
> Don't return directly `\\` in `to` (i.e `path\to\newFile`) option because on UNIX the backslash is a valid character inside a path component, i.e., it's not a separator.
> On Windows, the forward slash and the backward slash are both separators.
> Instead please use `/` or `path` methods.

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "src/*.png",
          to({ context, absoluteFilename }) {
            return "dest/newPath/[name][ext]";
          },
        },
      ],
    }),
  ],
};
```

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "src/*.png",
          to({ context, absoluteFilename }) {
            return Promise.resolve("dest/newPath/[name][ext]");
          },
        },
      ],
    }),
  ],
};
```

#### `context`

Type:

```ts
type context = string;
```

Default: `options.context|compiler.options.context`

A path to be (1) prepended to `from` and (2) removed from the start of the result path(s).

> **Warning**
>
> Don't use directly `\\` in `context` (i.e `path\to\context`) option because on UNIX the backslash is a valid character inside a path component, i.e., it's not a separator.
> On Windows, the forward slash and the backward slash are both separators.
> Instead please use `/` or `path` methods.

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "src/*.txt",
          to: "dest/",
          context: "app/",
        },
      ],
    }),
  ],
};
```

`context` can be an absolute path or a relative path. If it is a relative path, then it will be converted to an absolute path based on `compiler.options.context`.

`context` should be explicitly set only when `from` contains a glob. Otherwise, `context` is automatically set, based on whether `from` is a file or a directory:

If `from` is a file, then `context` is its directory. The result path will be the filename alone.

If `from` is a directory, then `context` equals `from`. The result paths will be the paths of the directory's contents (including nested contents), relative to the directory.

The use of `context` is illustrated by these [`examples`](#examples).

#### `globOptions`

Type:

```ts
type globOptions = import("globby").Options;
```

Default: `undefined`

Allows to configure the glob pattern matching library used by the plugin. [See the list of supported options][glob-options]
To exclude files from the selection, you should use [globOptions.ignore option](https://github.com/mrmlnc/fast-glob#ignore)

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "public/**/*",
          globOptions: {
            dot: true,
            gitignore: true,
            ignore: ["**/file.*", "**/ignored-directory/**"],
          },
        },
      ],
    }),
  ],
};
```

#### `filter`

Type:

```ts
type filter = (filepath: string) => boolean;
```

Default: `undefined`

> **Note**
>
> To ignore files by path please use the [`globOptions.ignore`](#globoptions) option.

**webpack.config.js**

```js
const fs = require("fs").promise;

module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "public/**/*",
          filter: async (resourcePath) => {
            const data = await fs.promises.readFile(resourcePath);
            const content = data.toString();

            if (content === "my-custom-content") {
              return false;
            }

            return true;
          },
        },
      ],
    }),
  ],
};
```

#### `toType`

Type:

```ts
type toType = "dir" | "file" | "template";
```

Default: `undefined`

Determinate what is `to` option - directory, file or template.
Sometimes it is hard to say what is `to`, example `path/to/dir-with.ext`.
If you want to copy files in directory you need use `dir` option.
We try to automatically determine the `type` so you most likely do not need this option.

|             Name              |   Type   |   Default   | Description                                                                                          |
| :---------------------------: | :------: | :---------: | :--------------------------------------------------------------------------------------------------- |
|      **[`'dir'`](#dir)**      | `string` | `undefined` | If `to` has no extension or ends on `'/'`                                                            |
|     **[`'file'`](#file)**     | `string` | `undefined` | If `to` is not a directory and is not a template                                                     |
| **[`'template'`](#template)** | `string` | `undefined` | If `to` contains [a template pattern](https://webpack.js.org/configuration/output/#template-strings) |

##### `'dir'`

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "path/to/file.txt",
          to: "directory/with/extension.ext",
          toType: "dir",
        },
      ],
    }),
  ],
};
```

##### `'file'`

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "path/to/file.txt",
          to: "file/without/extension",
          toType: "file",
        },
      ],
    }),
  ],
};
```

##### `'template'`

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "src/",
          to: "dest/[name].[contenthash][ext]",
          toType: "template",
        },
      ],
    }),
  ],
};
```

#### `force`

Type:

```ts
type force = boolean;
```

Default: `false`

Overwrites files already in `compilation.assets` (usually added by other plugins/loaders).

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "src/**/*",
          to: "dest/",
          force: true,
        },
      ],
    }),
  ],
};
```

#### `priority`

Type:

```ts
type priority = number;
```

Default: `0`

Allows to specify the priority of copying files with the same destination name.
Files for patterns with higher priority will be copied later.
To overwrite files, the [`force`](#force) option must be enabled.

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        // Copied second and will overwrite "dir_2/file.txt"
        {
          from: "dir_1/file.txt",
          to: "newfile.txt",
          force: true,
          priority: 10,
        },
        // Copied first
        {
          from: "dir_2/file.txt",
          to: "newfile.txt",
          priority: 5,
        },
      ],
    }),
  ],
};
```

#### `transform`

Type:

```ts
type transform =
  | {
      transformer: (input: string, absoluteFilename: string) => string | Buffer;
      cache?: boolean | TransformerCacheObject | undefined;
    }
  | ((input: string, absoluteFilename: string) => string | Buffer);
```

Default: `undefined`

Allows to modify the file contents.

##### `function`

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "src/*.png",
          to: "dest/",
          // The `content` argument is a [`Buffer`](https://nodejs.org/api/buffer.html) object, it could be converted to a `String` to be processed using `content.toString()`
          // The `absoluteFrom` argument is a `String`, it is absolute path from where the file is being copied
          transform(content, absoluteFrom) {
            return optimize(content);
          },
        },
      ],
    }),
  ],
};
```

##### `object`

|               Name                |   Default   | Description                                                                                                      |
| :-------------------------------: | :---------: | :--------------------------------------------------------------------------------------------------------------- |
| **[`transformer`](#transformer)** | `undefined` | Allows to modify the file contents.                                                                              |
|       **[`cache`](#cache)**       |   `false`   | Enable `transform` caching. You can use `transform: { cache: { key: 'my-cache-key' } }` to invalidate the cache. |

###### `transformer`

Type:

```ts
type transformer = (input: string, absoluteFilename: string) => string;
```

Default: `undefined`

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "src/*.png",
          to: "dest/",
          // The `content` argument is a [`Buffer`](https://nodejs.org/api/buffer.html) object, it could be converted to a `String` to be processed using `content.toString()`
          // The `absoluteFrom` argument is a `String`, it is absolute path from where the file is being copied
          transform: {
            transformer(content, absoluteFrom) {
              return optimize(content);
            },
          },
        },
      ],
    }),
  ],
};
```

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "src/*.png",
          to: "dest/",
          transform: {
            transformer(content, path) {
              return Promise.resolve(optimize(content));
            },
          },
        },
      ],
    }),
  ],
};
```

###### `cache`

Type:

```ts
type cache =
  | boolean
  | {
      keys: {
        [key: string]: any;
      };
    }
  | {
      keys: (
        defaultCacheKeys: {
          [key: string]: any;
        },
        absoluteFilename: string,
      ) => Promise<{
        [key: string]: any;
      }>;
    }
  | undefined;
```

Default: `false`

**webpack.config.js**

Enable/disable and configure caching.
Default path to cache directory: `node_modules/.cache/copy-webpack-plugin`.

###### `boolean`

Enables/Disable `transform` caching.

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "src/*.png",
          to: "dest/",
          transform: {
            transformer(content, path) {
              return optimize(content);
            },
            cache: true,
          },
        },
      ],
    }),
  ],
};
```

##### `object`

Enables `transform` caching and setup invalidation keys.

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "src/*.png",
          to: "dest/",
          transform: {
            transformer(content, path) {
              return optimize(content);
            },
            cache: {
              keys: {
                // May be useful for invalidating cache based on external values
                // For example, you can invalid cache based on `process.version` - { node: process.version }
                key: "value",
              },
            },
          },
        },
      ],
    }),
  ],
};
```

You can setup invalidation keys using a function.

Simple function:

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "src/*.png",
          to: "dest/",
          transform: {
            transformer(content, path) {
              return optimize(content);
            },
            cache: {
              keys: (defaultCacheKeys, absoluteFrom) => {
                const keys = getCustomCacheInvalidationKeysSync();

                return {
                  ...defaultCacheKeys,
                  keys,
                };
              },
            },
          },
        },
      ],
    }),
  ],
};
```

Async function:

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "src/*.png",
          to: "dest/",
          transform: {
            transformer(content, path) {
              return optimize(content);
            },
            cache: {
              keys: async (defaultCacheKeys, absoluteFrom) => {
                const keys = await getCustomCacheInvalidationKeysAsync();

                return {
                  ...defaultCacheKeys,
                  keys,
                };
              },
            },
          },
        },
      ],
    }),
  ],
};
```

#### `transformAll`

Type:

```ts
type transformAll = (
  data: {
    data: Buffer;
    sourceFilename: string;
    absoluteFilename: string;
  }[],
) => any;
```

Default: `undefined`

Allows you to modify the contents of multiple files and save the result to one file.

> **Note**
>
> The `to` option must be specified and point to a file. It is allowed to use only `[contenthash]` and `[fullhash]` template strings.

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "src/**/*.txt",
          to: "dest/file.txt",
          // The `assets` argument is an assets array for the pattern.from ("src/**/*.txt")
          transformAll(assets) {
            const result = assets.reduce((accumulator, asset) => {
              // The asset content can be obtained from `asset.source` using `source` method.
              // The asset content is a [`Buffer`](https://nodejs.org/api/buffer.html) object, it could be converted to a `String` to be processed using `content.toString()`
              const content = asset.data;

              accumulator = `${accumulator}${content}\n`;
              return accumulator;
            }, "");

            return result;
          },
        },
      ],
    }),
  ],
};
```

### `noErrorOnMissing`

Type:

```ts
type noErrorOnMissing = boolean;
```

Default: `false`

Doesn't generate an error on missing file(s).

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: path.resolve(__dirname, "missing-file.txt"),
          noErrorOnMissing: true,
        },
      ],
    }),
  ],
};
```

#### `info`

Type:

```ts
type info =
  | Record<string, any>
  | ((item: {
      absoluteFilename: string;
      sourceFilename: string;
      filename: string;
      toType: ToType;
    }) => Record<string, any>);
```

Default: `undefined`

Allows to add assets info.

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        "relative/path/to/file.ext",
        {
          from: "**/*",
          // Terser skip this file for minimization
          info: { minimized: true },
        },
      ],
    }),
  ],
};
```

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        "relative/path/to/file.ext",
        {
          from: "**/*",
          // Terser skip this file for minimization
          info: (file) => ({ minimized: true }),
        },
      ],
    }),
  ],
};
```

### Options

- [`concurrency`](#concurrency)

#### `concurrency`

type:

```ts
type concurrency = number;
```

default: `100`

limits the number of simultaneous requests to fs

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [...patterns],
      options: { concurrency: 50 },
    }),
  ],
};
```

### Examples

#### Different variants of `from` (`glob`, `file` or `dir`).

Take for example the following file structure:

```
src/directory-nested/deep-nested/deepnested-file.txt
src/directory-nested/nested-file.txt
```

##### From is a Glob

Everything that you specify in `from` will be included in the result:

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "src/directory-nested/**/*",
        },
      ],
    }),
  ],
};
```

Result:

```txt
src/directory-nested/deep-nested/deepnested-file.txt,
src/directory-nested/nested-file.txt
```

If you don't want the result paths to start with `src/directory-nested/`, then you should move `src/directory-nested/` to `context`, such that only the glob pattern `**/*` remains in `from`:

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "**/*",
          context: path.resolve(__dirname, "src", "directory-nested"),
        },
      ],
    }),
  ],
};
```

Result:

```txt
deep-nested/deepnested-file.txt,
nested-file.txt
```

##### From is a Dir

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: path.resolve(__dirname, "src", "directory-nested"),
        },
      ],
    }),
  ],
};
```

Result:

```txt
deep-nested/deepnested-file.txt,
nested-file.txt
```

Technically, this is `**/*` with a predefined context equal to the specified directory.

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "**/*",
          context: path.resolve(__dirname, "src", "directory-nested"),
        },
      ],
    }),
  ],
};
```

Result:

```txt
deep-nested/deepnested-file.txt,
nested-file.txt
```

##### From is a File

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: path.resolve(
            __dirname,
            "src",
            "directory-nested",
            "nested-file.txt",
          ),
        },
      ],
    }),
  ],
};
```

Result:

```txt
nested-file.txt
```

Technically, this is a filename with a predefined context equal to `path.dirname(pathToFile)`.

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "nested-file.txt",
          context: path.resolve(__dirname, "src", "directory-nested"),
        },
      ],
    }),
  ],
};
```

Result:

```txt
nested-file.txt
```

#### Ignoring files

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: path.posix.join(
            path.resolve(__dirname, "src").replace(/\\/g, "/"),
            "**/*",
          ),
          globOptions: {
            ignore: [
              // Ignore all `txt` files
              "**/*.txt",
              // Ignore all files in all subdirectories
              "**/subdir/**",
            ],
          },
        },
      ],
    }),
  ],
};
```

#### Flatten copy

Removes all directory references and only copies file names.

> **Warning**
>
> If files have the same name, the result is non-deterministic.

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "src/**/*",
          to: "[name][ext]",
        },
      ],
    }),
  ],
};
```

Result:

```txt
file-1.txt
file-2.txt
nested-file.txt
```

#### Copy in new directory

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          // When copying files starting with a dot, must specify the toType option
          // toType: "file",
          to({ context, absoluteFilename }) {
            return `newdirectory/${path.relative(context, absoluteFilename)}`;
          },
          from: "directory",
        },
      ],
    }),
  ],
};
```

Result:

```txt
"newdirectory/file-1.txt",
"newdirectory/nestedfile.txt",
"newdirectory/nested/deep-nested/deepnested.txt",
"newdirectory/nested/nestedfile.txt",
```

#### Skip running JavaScript files through a minimizer

Useful if you need to simply copy `*.js` files to destination "as is" without evaluating and minimizing them using Terser.

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        "relative/path/to/file.ext",
        {
          from: "**/*",
          // Terser skip this file for minimization
          info: { minimized: true },
        },
      ],
    }),
  ],
};
```

##### `yarn workspaces` and `monorepos`

When using `yarn workspaces` or` monorepos`, relative copy paths from node_modules can be broken due to the way packages are hoisting.
To avoid this, should explicitly specify where to copy the files from using `require.resolve`.

**webpack.config.js**

```js
module.exports = {
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: `${path.dirname(
            require.resolve(`${moduleName}/package.json`),
          )}/target`,
          to: "target",
        },
      ],
    }),
  ],
};
```

## Contributing

Please take a moment to read our contributing guidelines if you haven't yet done so.

[CONTRIBUTING](./.github/CONTRIBUTING.md)

## License

[MIT](./LICENSE)

[npm]: https://img.shields.io/npm/v/copy-webpack-plugin.svg
[npm-url]: https://npmjs.com/package/copy-webpack-plugin
[node]: https://img.shields.io/node/v/copy-webpack-plugin.svg
[node-url]: https://nodejs.org
[tests]: https://github.com/webpack-contrib/copy-webpack-plugin/workflows/copy-webpack-plugin/badge.svg
[tests-url]: https://github.com/webpack-contrib/copy-webpack-plugin/actions
[cover]: https://codecov.io/gh/webpack-contrib/copy-webpack-plugin/branch/master/graph/badge.svg
[cover-url]: https://codecov.io/gh/webpack-contrib/copy-webpack-plugin
[discussion]: https://img.shields.io/github/discussions/webpack/webpack
[discussion-url]: https://github.com/webpack/webpack/discussions
[size]: https://packagephobia.now.sh/badge?p=copy-webpack-plugin
[size-url]: https://packagephobia.now.sh/result?p=copy-webpack-plugin
[glob-options]: https://github.com/sindresorhus/globby#options
