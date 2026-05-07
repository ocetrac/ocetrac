{% if fullname.startswith('ocetrac.') %}
{{ fullname[8:] | underline}}
{% else %}
{{ fullname | underline}}
{% endif %}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
   :members:
   :inherited-members: