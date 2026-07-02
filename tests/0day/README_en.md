# Models with Day-Zero Launch Support on Ascend

Certification `[Pass]` indicates that a model has passed testing on the official Ascend release, and `Test` indicates a model awaiting testing.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Download Link</th>
      <th>Script Path</th>
      <th>Sequence</th>
      <th>Implementation</th>
      <th>Cluster</th>
      <th>Contributor</th>
      <th>Certification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"> <a href="https://modelscope.cn/collections/GLM-4-0414-e4ecc89c179d4c">GLM-4</a> </td>
      <td><a href="https://modelscope.cn/models/ZhipuAI/GLM-4-9B-0414">9B</a></td>
      <td><a href="glm4-9b-0414">GLM-4-9B-0414</a></td>
      <td> 8K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td><a href="https://modelscope.cn/models/ZhipuAI/GLM-4-32B-0414">32B</a></td>
      <td><a href="glm4-32b-0414">GLM-4-32B-0414</a></td>
      <td> 8K </td>
      <th> MCore </th>
      <td> 4x8 </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td><a href="https://modelscope.cn/models/ZhipuAI/GLM-4-32B-Base-0414">32B-Base</a></td>
      <td><a href="glm4-base-32b-0414">GLM-4-base-32B-0414</a></td>
      <td> 8K </td>
      <th> MCore </th>
      <td> 4x8 </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="3"> <a href="https://modelscope.cn/collections/GLM-4-0414-e4ecc89c179d4c">GLM-Z1</a> </td>
      <td><a href="https://modelscope.cn/models/ZhipuAI/GLM-Z1-9B-0414">9B</a></td>
      <td><a href="glm-z1-9b-0414">GLM-Z1-9B-0414</a></td>
      <td> 8K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td><a href="https://modelscope.cn/models/ZhipuAI/GLM-Z1-32B-0414">32B</a></td>
      <td><a href="glm-z1-32b-0414">GLM-Z1-32B-0414</a></td>
      <td> 8K </td>
      <th> MCore </th>
      <td> 4x8 </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td><a href="https://modelscope.cn/models/ZhipuAI/GLM-Z1-Rumination-32B-0414">Rumination-32B</a></td>
      <td><a href="glm-z1-rumination-32b-0414">GLM-Z1-Rumination-32B-0414</a></td>
      <td> 8K </td>
      <th> MCore </th>
      <td> 4x8 </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
  </tbody>
</table>

## Model Script Notes

These models currently support only the basic functionality required for the 0day first release. They are in internal testing and have not completed sufficient performance testing or acceptance testing. Undiscovered issues may still remain during actual use. Therefore, the official version will be released after sufficient validation.

## Version Mapping

The dependencies for these models are listed in the following table.

<table>
  <tr>
    <th>Dependencies</th>
    <th>Version</th>
  </tr>
  <tr>
    <td>Ascend NPU driver</td>
    <td rowspan="2">Under development</td>
  </tr>
  <tr>
    <td>Ascend NPU firmware</td>
  </tr>
  <tr>
    <td>Toolkit (development kit)</td>
    <td rowspan="3">Under development</td>
  </tr>
  <tr>
    <td>Kernel (operator package)</td>
  </tr>
  <tr>
    <td>Ascend Transformer Boost acceleration library (NNAL)</td>
  </tr>
  <tr>
    <td>Python</td>
    <td>3.10</td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td rowspan="3">2.5</td>
  </tr>
  <tr>
    <td>torch_npu plugin</td>
  </tr>
  <tr>
    <td>apex</td>
  </tr>
  <tr>
    <td>transformers</td>
    <td>4.51.3</td>
  </tr>
</table>

## Version Notes

### Reference Implementation

```shell
url=https://github.com/huggingface/transformers/tree/v4.51.3
commit_id=5f4ecf2
```

### Change Log

- 2025-04-22: Initial release.
