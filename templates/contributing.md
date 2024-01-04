# On Github Issues and Pull Requests

Found a bug? Have a new feature to suggest?
Want to add a new code examples to keras.io, or to contribute changes to the codebase?
Make sure to read this first.

---

## Bug reporting

Your code doesn't work, and you have determined that the issue lies with Keras? Follow these steps to report a bug.

1. Your bug may already be fixed. Make sure to update to the current Keras nightly release (`pip install keras-nightly --upgrade`) and test whether your bug is still occurring.

2. Search for similar issues among the [keras-team/keras Github issues](https://github.com/keras-team/keras/issues). Make sure to delete `is:open` on the issue search to find solved tickets as well. It's possible somebody has encountered this bug already. Also remember to check out Keras [FAQ](http://keras.io/getting_started/faq/). Still having a problem? Open an issue on the Keras Github to let us know.

3. Make sure you provide us with useful information about your configuration: What backend are you using? Are you running on GPU?

4. Provide us with a standalone script to reproduce the issue. This script **should be runnable as-is** and should not require external data download (use randomly generated data if you need to run a model on some test data). We recommend that you use GitHub Gists or Colab to post your code. Any issue that cannot be reproduced is likely to be closed.

5. If possible, take a stab at fixing the bug yourself -- if you can!

The more information you provide, the easier it is for us to validate that there is a bug and the faster we'll be able to take action. If you want your issue to be resolved quickly, following the steps above is crucial.

---

## Requesting a Feature

You can use [keras-team/keras Github issues](https://github.com/keras-team/keras/issues) to request features you would like to see in Keras, or changes in the Keras API.

1. Provide a clear and detailed explanation of the feature you want and why it's important to add. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing an add-on library for Keras. It is crucial for Keras to avoid bloating the API and codebase.

2. Provide code snippets demonstrating the API you have in mind and illustrating the use cases of your feature. Of course, you don't need to write any real code at this point!


---

## Submitting a Pull Request

You can open a PR at [keras-team/keras](https://github.com/keras-team/keras/pulls). 

Please note that PRs that are primarily about **code style** (as opposed to fixing bugs, improving docs, or adding new functionality) will likely be rejected.

---

## Adding new examples

Even if you don't contribute to the Keras source code, if you have an application of Keras that is concise and powerful,
please consider adding it to our collection of examples, featured on [keras.io](https://keras.io). 
[Follow these steps](/examples/#adding-a-new-code-example) to submit a new code examples.
